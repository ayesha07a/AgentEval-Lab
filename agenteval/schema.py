from __future__ import annotations

from dataclasses import dataclass, field

CATEGORIES = {"factual", "multi_hop", "reasoning", "unanswerable"}


@dataclass(frozen=True)
class Evidence:
    source: str  # corpus file name, e.g. "helio_fares.md"
    quote: str   # sentence that must appear verbatim in that file


@dataclass(frozen=True)
class TestCase:
    __test__ = False  # stop pytest from collecting this as a test class

    id: str
    question: str
    category: str
    answerable: bool
    expected_answer: str | None  # None when the question is unanswerable
    evidence: list[Evidence] = field(default_factory=list)
    # Facts an answer must contain to count as correct. Each item may list alternatives
    # separated by '|' (e.g. "all|three"). Used by the deterministic scorer.
    key_facts: list[str] = field(default_factory=list)
