import json
from pathlib import Path

from agenteval.agent import Agent, build_prompt
from agenteval.dataset import load_corpus, load_dataset
from agenteval.llm import MockClient
from agenteval.retrieval import Passage, Retriever
from agenteval.run import run_agent, write_results

ROOT = Path(__file__).resolve().parent.parent
CORPUS = load_corpus(ROOT / "data" / "corpus")
CASES = load_dataset(ROOT / "data" / "eval_set.jsonl")


def test_prompt_contains_passages_and_question():
    p = build_prompt("Q?", [Passage("a.md", "Fact one.")])
    assert "[1] (a.md) Fact one." in p and "Question: Q?" in p


def test_agent_returns_client_answer():
    agent = Agent(MockClient("  42  "), Retriever(CORPUS))
    r = agent.answer("How many stations does the Harbor Line have?")
    assert r.answer == "42" and r.error is None and r.passages


def test_agent_records_errors_instead_of_raising():
    class Broken:
        def complete(self, prompt, system=None):
            raise RuntimeError("boom")

    r = Agent(Broken(), Retriever(CORPUS)).answer("anything about fares")
    assert r.answer == "" and "boom" in r.error


def test_run_writes_one_row_per_case(tmp_path):
    rows = run_agent(Agent(MockClient(), Retriever(CORPUS)), CASES, {"run_id": "t"})
    out = tmp_path / "r.jsonl"
    write_results(rows, out)
    lines = out.read_text().splitlines()
    assert len(lines) == len(CASES)
    assert json.loads(lines[0])["run_id"] == "t"
