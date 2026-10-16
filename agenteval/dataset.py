from __future__ import annotations

import json
from pathlib import Path

from .schema import CATEGORIES, Evidence, TestCase


def load_corpus(corpus_dir: str | Path) -> dict[str, str]:
    """Map file name -> text for every .md file in the corpus directory."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(corpus_dir).glob("*.md"))}


def load_dataset(path: str | Path) -> list[TestCase]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            cases.append(
                TestCase(
                    id=raw["id"],
                    question=raw["question"],
                    category=raw["category"],
                    answerable=raw["answerable"],
                    expected_answer=raw["expected_answer"],
                    evidence=[Evidence(**e) for e in raw.get("evidence", [])],
                    key_facts=raw.get("key_facts", []),
                )
            )
    return cases


def validate_dataset(cases: list[TestCase], corpus: dict[str, str]) -> list[str]:
    """Return a list of problems; an empty list means the dataset is valid."""
    errors: list[str] = []
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            errors.append(f"{c.id}: duplicate id")
        seen.add(c.id)
        if c.category not in CATEGORIES:
            errors.append(f"{c.id}: unknown category '{c.category}'")
        if c.answerable:
            if not c.expected_answer:
                errors.append(f"{c.id}: answerable case needs an expected_answer")
            if not c.evidence:
                errors.append(f"{c.id}: answerable case needs evidence")
            if not c.key_facts:
                errors.append(f"{c.id}: answerable case needs key_facts")
        else:
            if c.expected_answer is not None:
                errors.append(f"{c.id}: unanswerable case must have expected_answer null")
            if c.evidence:
                errors.append(f"{c.id}: unanswerable case must have no evidence")
            if c.key_facts:
                errors.append(f"{c.id}: unanswerable case must have no key_facts")
            if c.category != "unanswerable":
                errors.append(f"{c.id}: unanswerable case must use category 'unanswerable'")
        for e in c.evidence:
            if e.source not in corpus:
                errors.append(f"{c.id}: evidence source '{e.source}' not in corpus")
            elif e.quote not in corpus[e.source]:
                errors.append(f"{c.id}: evidence quote not found verbatim in {e.source}")
    return errors
