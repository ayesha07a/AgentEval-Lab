from pathlib import Path

from agenteval.dataset import load_corpus, load_dataset
from agenteval.retrieval import Retriever, retrieval_recall, split_sentences, tokenize

ROOT = Path(__file__).resolve().parent.parent
CORPUS = load_corpus(ROOT / "data" / "corpus")
CASES = load_dataset(ROOT / "data" / "eval_set.jsonl")


def test_tokenize_keeps_decimals_and_drops_stopwords():
    assert tokenize("What is the fare of 2.40 credits?") == ["fare", "2.40", "credits"]


def test_split_sentences_skips_headings():
    out = split_sentences("# Title\n\nOne fact. Two facts.")
    assert out == ["One fact.", "Two facts."]


def test_top_result_for_clear_query():
    top = Retriever(CORPUS).search("How much is a single adult fare?", k=1)[0]
    assert "2.40" in top.text


def test_no_overlap_returns_nothing():
    assert Retriever(CORPUS).search("zzzz qqqq", k=3) == []


def test_recall_is_a_valid_fraction():
    r = retrieval_recall(CASES, Retriever(CORPUS), k=3)
    assert 0.0 <= r <= 1.0
