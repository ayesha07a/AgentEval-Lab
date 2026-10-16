from pathlib import Path

from agenteval.dataset import load_corpus, load_dataset, validate_dataset
from agenteval.llm import MockClient
from agenteval.schema import Evidence, TestCase

ROOT = Path(__file__).resolve().parent.parent
CORPUS = load_corpus(ROOT / "data" / "corpus")
CASES = load_dataset(ROOT / "data" / "eval_set.jsonl")


def make_case(**overrides):
    base = dict(
        id="t1", question="q?", category="factual", answerable=True,
        expected_answer="a", evidence=[Evidence("helio_fares.md", "Children under 6 ride free.")],
        key_facts=["free"],
    )
    base.update(overrides)
    return TestCase(**base)


def test_seed_dataset_is_valid():
    assert validate_dataset(CASES, CORPUS) == []


def test_seed_dataset_includes_unanswerable_cases():
    assert any(not c.answerable for c in CASES)


def test_duplicate_ids_detected():
    errors = validate_dataset([make_case(), make_case()], CORPUS)
    assert any("duplicate id" in e for e in errors)


def test_missing_evidence_quote_detected():
    bad = make_case(evidence=[Evidence("helio_fares.md", "This sentence is not in the corpus.")])
    assert any("not found verbatim" in e for e in validate_dataset([bad], CORPUS))


def test_unknown_source_detected():
    bad = make_case(evidence=[Evidence("nope.md", "x")])
    assert any("not in corpus" in e for e in validate_dataset([bad], CORPUS))


def test_unanswerable_with_answer_detected():
    bad = make_case(answerable=False, category="unanswerable", evidence=[])
    assert any("expected_answer null" in e for e in validate_dataset([bad], CORPUS))


def test_mock_client_returns_response():
    r = MockClient(lambda p: p.upper()).complete("hi")
    assert r.text == "HI" and r.latency_s >= 0


def test_answerable_without_key_facts_detected():
    errors = validate_dataset([make_case(key_facts=[])], CORPUS)
    assert any("needs key_facts" in e for e in errors)
