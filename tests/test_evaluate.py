import json
from pathlib import Path

from agenteval.agent import Agent
from agenteval.dataset import load_corpus, load_dataset
from agenteval.evaluate import main, score_results, summarize
from agenteval.llm import MockClient
from agenteval.retrieval import Retriever
from agenteval.run import run_agent, write_results

ROOT = Path(__file__).resolve().parent.parent
CORPUS = load_corpus(ROOT / "data" / "corpus")
CASES = load_dataset(ROOT / "data" / "eval_set.jsonl")
EXPECTED = {c.question: c for c in CASES}


def _question(prompt: str) -> str:
    return prompt.split("Question: ")[1].split("\n")[0]


def _run(responder):
    agent = Agent(MockClient(responder), Retriever(CORPUS))
    return run_agent(agent, CASES, {"run_id": "t"})


def oracle(prompt):  # perfect agent: reference answer, or abstain when unanswerable
    c = EXPECTED[_question(prompt)]
    return c.expected_answer if c.answerable else "I don't know; the context does not say."


def confident_liar(prompt):  # always answers, with a fabricated figure
    return "It is 99 credits."


def test_perfect_agent_scores_perfectly():
    s = summarize(score_results(_run(oracle), CASES))
    assert s["accuracy"] == 1.0
    assert s["hallucination_rate"] == 0.0
    assert s["groundedness_rate"] == 1.0
    assert s["wrong_refusal_rate"] == 0.0


def test_fabricating_agent_is_caught():
    s = summarize(score_results(_run(confident_liar), CASES))
    assert s["answerable_accuracy"] == 0.0
    assert s["unanswerable_abstention_rate"] == 0.0
    assert s["hallucination_rate"] == 1.0


def test_always_refusing_agent_is_not_accurate_on_answerable():
    s = summarize(score_results(_run(lambda p: "I don't know."), CASES))
    assert s["answerable_accuracy"] == 0.0 and s["wrong_refusal_rate"] == 1.0
    assert s["unanswerable_abstention_rate"] == 1.0 and s["hallucination_rate"] == 0.0


def test_errors_are_excluded_from_rates():
    class Broken:
        def complete(self, prompt, system=None):
            raise RuntimeError("down")

    rows = run_agent(Agent(Broken(), Retriever(CORPUS)), CASES, {})
    s = summarize(score_results(rows, CASES))
    assert s["n_errors"] == len(CASES) and s["n_scored"] == 0 and s["accuracy"] is None


def test_cli_writes_scored_csv_summary_and_report(tmp_path):
    results = tmp_path / "base.jsonl"
    write_results(_run(oracle), results)
    main(["--results", str(results)])
    assert (tmp_path / "base_scored.csv").exists()
    assert json.loads((tmp_path / "base_summary.json").read_text())["accuracy"] == 1.0
    assert "Metric definitions" in (tmp_path / "base_report.md").read_text()
