"""Run the agent over the eval set and save raw results (no scoring yet; that is Phase 3).

Usage:
  python -m agenteval.run --provider mock --out reports/mock.jsonl
  python -m agenteval.run --provider anthropic --model <model-name> --out reports/baseline.jsonl
"""
from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .agent import PROMPT_VERSION, Agent
from .clients import make_client
from .dataset import load_corpus, load_dataset, validate_dataset
from .retrieval import Retriever
from .schema import TestCase

ROOT = Path(__file__).resolve().parent.parent


def run_agent(agent: Agent, cases: list[TestCase], meta: dict) -> list[dict]:
    rows = []
    for c in cases:
        r = agent.answer(c.question)
        rows.append({
            **meta,
            "id": c.id, "question": c.question, "category": c.category,
            "answerable": c.answerable, "expected_answer": c.expected_answer,
            "answer": r.answer, "retrieved": [asdict(p) for p in r.passages],
            "latency_s": r.latency_s, "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens, "error": r.error,
        })
    return rows


def write_results(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "anthropic", "openai-compat"])
    ap.add_argument("--model")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "reports" / "run.jsonl"))
    args = ap.parse_args(argv)

    corpus = load_corpus(ROOT / "data" / "corpus")
    cases = load_dataset(ROOT / "data" / "eval_set.jsonl")
    problems = validate_dataset(cases, corpus)
    if problems:
        raise SystemExit("Dataset invalid:\n" + "\n".join(problems))

    agent = Agent(make_client(args.provider, args.model), Retriever(corpus), k=args.k)
    meta = {
        "run_id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": args.provider, "model": args.model, "k": args.k,
        "prompt_version": PROMPT_VERSION,
    }
    rows = run_agent(agent, cases, meta)
    write_results(rows, args.out)
    errors = sum(1 for r in rows if r["error"])
    print(f"Wrote {len(rows)} rows to {args.out} ({errors} errors), run_id={meta['run_id']}")


if __name__ == "__main__":
    main()
