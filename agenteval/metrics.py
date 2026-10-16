"""Deterministic scoring helpers. These are heuristics: cheap, repeatable, and
transparent, but not a substitute for an LLM judge or human labels (see judge.py)."""
from __future__ import annotations

import re
from itertools import combinations

from .retrieval import tokenize as content_tokens

_WORD = re.compile(r"\d+(?:\.\d+)?|[a-z]+")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_NEUTRAL = {"yes", "no", "not", "because", "therefore", "so", "only", "also", "within", "falls"}

_ABSTAIN = re.compile(
    r"don'?t know|do not know|no information|not (?:mentioned|specified|stated|provided|"
    r"available|found|included|covered)|(?:cannot|can'?t|unable to) (?:be )?(?:determine|answer|find|say|tell)|"
    r"insufficient|context (?:does not|doesn'?t)|unanswerable|no (?:data|mention)",
    re.IGNORECASE,
)


def norm_num(token: str) -> str:
    try:
        f = float(token)
    except ValueError:
        return token
    return str(int(f)) if f == int(f) else str(round(f, 6))


def words(text: str) -> set[str]:
    return {norm_num(t) for t in _WORD.findall(text.lower())}


def numbers(text: str) -> set[str]:
    return {norm_num(n) for n in _NUM.findall(text)}


def is_abstention(answer: str) -> bool:
    return bool(_ABSTAIN.search(answer))


def fact_present(fact: str, answer_words: set[str]) -> bool:
    """True if any '|'-separated alternative has all its words in the answer."""
    for alt in fact.split("|"):
        toks = [norm_num(t) for t in _WORD.findall(alt.lower())]
        if toks and all(t in answer_words for t in toks):
            return True
    return False


def heuristic_correct(answer: str, key_facts: list[str]) -> bool:
    if not answer.strip() or not key_facts:
        return False
    aw = words(answer)
    return all(fact_present(f, aw) for f in key_facts)


def is_grounded(answer: str, passages: list[str], question: str = "", threshold: float = 0.7) -> bool:
    """Heuristic groundedness: every number in the answer must appear in the context or
    question (or be a difference of two such numbers), and most content words must too."""
    context = " ".join(passages)
    base = {float(n) for n in _NUM.findall(context + " " + question)}
    allowed = {norm_num(str(n)) for n in base}
    allowed |= {norm_num(str(round(abs(a - b), 6))) for a, b in combinations(base, 2)}
    if any(n not in allowed for n in numbers(answer)):
        return False
    known = set(content_tokens(context + " " + question))
    toks = [t for t in content_tokens(answer) if not _NUM.fullmatch(t) and t not in _NEUTRAL]
    if not toks:
        return True
    return sum(t in known for t in toks) / len(toks) >= threshold
