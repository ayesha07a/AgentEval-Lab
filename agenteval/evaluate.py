"""Score a saved run (from agenteval.run) and write a CSV, summary JSON, and report.

Usage:
  python -m agenteval.evaluate --results reports/baseline.jsonl
  python -m agenteval.evaluate --results reports/baseline.jsonl \\
      --judge-provider anthropic --judge-model <model-name>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .clients import make_client
from .dataset import load_dataset
from .judge import LLMJudge
from .metrics import heuristic_correct, is_abstention, is_grounded
from .schema import TestCase

ROOT = Path(__file__).resolve().parent.parent


def load_results(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_row(row: dict, case: TestCase, judge: LLMJudge | None = None) -> dict:
    base = {
        "id": row["id"], "category": case.category, "answerable": case.answerable,
        "question": case.question, "expected_answer": case.expected_answer,
        "answer": row.get("answer", ""), "latency_s": row.get("latency_s"),
        "input_tokens": row.get("input_tokens"), "output_tokens": row.get("output_tokens"),
        "error": row.get("error"),
    }
    if row.get("error"):  # API failures are reported, never scored as wrong answers
        return {**base, "scored": False, "abstained": None, "correct": None, "grounded": None,
                "hallucinated": None, "wrongly_refused": None, "scored_by": None}

    answer = row.get("answer", "") or ""
    passages = [p["text"] for p in row.get("retrieved", [])]
    abstained = is_abstention(answer)
    used_judge = False

    if case.answerable:
        correct = False
        if not abstained:
            verdict = judge.correct(case.question, case.expected_answer, answer) if judge else None
            if verdict is None:
                verdict = heuristic_correct(answer, case.key_facts)
            else:
                used_judge = True
            correct = verdict
    else:
        correct = abstained  # the only right behavior is to decline

    grounded = None
    if not abstained:
        g = judge.grounded(answer, passages) if judge else None
        if g is None:
            g = is_grounded(answer, passages, case.question)
        else:
            used_judge = True
        grounded = g

    hallucinated = (not abstained) and ((grounded is False) or (not case.answerable))
    return {
        **base, "scored": True, "abstained": abstained, "correct": correct, "grounded": grounded,
        "hallucinated": hallucinated, "wrongly_refused": case.answerable and abstained,
        "scored_by": "judge" if used_judge else "heuristic",
    }


def score_results(rows: list[dict], cases: list[TestCase], judge: LLMJudge | None = None) -> pd.DataFrame:
    by_id = {c.id: c for c in cases}
    missing = [r["id"] for r in rows if r["id"] not in by_id]
    if missing:
        raise ValueError(f"results contain ids not in the dataset: {missing}")
    return pd.DataFrame([score_row(r, by_id[r["id"]], judge) for r in rows])


def _rate(series: pd.Series) -> float | None:
    s = series.dropna()
    return float(s.astype(bool).mean()) if len(s) else None


def summarize(df: pd.DataFrame) -> dict:
    scored = df[df["scored"]]
    ans, unans = scored[scored["answerable"]], scored[~scored["answerable"]]
    answered = scored[scored["abstained"] == False]  # noqa: E712
    lat = scored["latency_s"].dropna()
    return {
        "n_total": int(len(df)), "n_errors": int((~df["scored"]).sum()), "n_scored": int(len(scored)),
        "accuracy": _rate(scored["correct"]),
        "answerable_accuracy": _rate(ans["correct"]),
        "unanswerable_abstention_rate": _rate(unans["correct"]),
        "hallucination_rate": _rate(scored["hallucinated"]),
        "groundedness_rate": _rate(answered["grounded"]),
        "wrong_refusal_rate": _rate(ans["wrongly_refused"]),
        "latency_mean_s": float(lat.mean()) if len(lat) else None,
        "latency_p50_s": float(lat.median()) if len(lat) else None,
        "latency_p95_s": float(lat.quantile(0.95)) if len(lat) else None,
        "input_tokens_total": int(scored["input_tokens"].fillna(0).sum()),
        "output_tokens_total": int(scored["output_tokens"].fillna(0).sum()),
        "accuracy_by_category": {k: float(v) for k, v in scored.groupby("category")["correct"].mean().items()},
        "scored_by": scored["scored_by"].value_counts().to_dict(),
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


DEFINITIONS = """\
- **Accuracy**: share of scored questions handled correctly. Answerable: the answer states the reference facts. Unanswerable: the agent declines to answer.
- **Hallucination rate**: share of scored questions where the agent gave an answer that is unsupported by the retrieved context, or answered a question that has no answer in the corpus.
- **Groundedness**: among non-abstaining answers, the share whose claims are supported by the retrieved context.
- **Wrong refusal rate**: among answerable questions, the share where the agent declined.
- Errors (e.g. API failures) are excluded from every rate and reported separately."""


def build_report(summary: dict, df: pd.DataFrame, meta: dict) -> str:
    s = summary
    lines = [
        "# Evaluation report", "",
        f"Run `{meta.get('run_id')}` | model: {meta.get('model')} | prompt: {meta.get('prompt_version')} | k={meta.get('k')}",
        f"Scoring: {s['scored_by'] or 'n/a'}", "",
        "| Metric | Value |", "|---|---|",
        f"| Questions scored | {s['n_scored']} of {s['n_total']} ({s['n_errors']} errors) |",
        f"| Accuracy | {_pct(s['accuracy'])} |",
        f"| Answerable accuracy | {_pct(s['answerable_accuracy'])} |",
        f"| Unanswerable abstention rate | {_pct(s['unanswerable_abstention_rate'])} |",
        f"| Hallucination rate | {_pct(s['hallucination_rate'])} |",
        f"| Groundedness | {_pct(s['groundedness_rate'])} |",
        f"| Wrong refusal rate | {_pct(s['wrong_refusal_rate'])} |",
        f"| Latency p50 / p95 | {s['latency_p50_s'] or 0:.2f}s / {s['latency_p95_s'] or 0:.2f}s |", "",
        "## Metric definitions", DEFINITIONS, "", "## Failures", "",
    ]
    fails = df[df["scored"] & ((df["correct"] == False) | (df["hallucinated"] == True))]  # noqa: E712
    if fails.empty:
        lines.append("None.")
    for _, r in fails.iterrows():
        lines.append(f"- `{r['id']}` ({r['category']}): {r['question']}\n  - answer: {r['answer']!r}\n  - expected: {r['expected_answer']!r}"
                     f" | hallucinated={r['hallucinated']} wrongly_refused={r['wrongly_refused']}")
    lines += ["", "## Caveats",
              f"Small dataset ({s['n_total']} questions): treat differences of a few points as noise. "
              "Heuristic scoring is approximate; spot-check scored rows by hand before trusting a number."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--dataset", default=str(ROOT / "data" / "eval_set.jsonl"))
    ap.add_argument("--out-dir", default=None, help="default: next to the results file")
    ap.add_argument("--judge-provider", choices=["anthropic", "openai-compat"])
    ap.add_argument("--judge-model")
    args = ap.parse_args(argv)

    rows = load_results(args.results)
    judge = LLMJudge(make_client(args.judge_provider, args.judge_model)) if args.judge_provider else None
    df = score_results(rows, load_dataset(args.dataset), judge)
    summary = summarize(df)

    out = Path(args.out_dir) if args.out_dir else Path(args.results).parent
    stem = Path(args.results).stem
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{stem}_scored.csv", index=False)
    (out / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    meta = {k: rows[0].get(k) for k in ("run_id", "model", "prompt_version", "k")} if rows else {}
    (out / f"{stem}_report.md").write_text(build_report(summary, df, meta))
    print(f"accuracy={_pct(summary['accuracy'])} hallucination={_pct(summary['hallucination_rate'])} "
          f"groundedness={_pct(summary['groundedness_rate'])} -> {out}")


if __name__ == "__main__":
    main()
