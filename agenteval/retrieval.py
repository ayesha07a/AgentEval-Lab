"""Sentence-level BM25 retrieval over the corpus (pure Python, no dependencies)."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .schema import TestCase

_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")
_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "is", "are", "was", "were", "do", "does",
    "did", "and", "or", "for", "by", "with", "it", "its", "that", "this", "what", "which",
    "who", "how", "can", "i", "be", "from", "as", "have", "has", "had",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Passage:
    source: str
    text: str
    score: float = 0.0


def split_sentences(text: str) -> list[str]:
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sentences.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    return sentences


class Retriever:
    def __init__(self, corpus: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.passages = [
            Passage(source=name, text=s) for name, text in corpus.items() for s in split_sentences(text)
        ]
        self._tokens = [tokenize(p.text) for p in self.passages]
        self._tf = [Counter(t) for t in self._tokens]
        self._avg_len = sum(len(t) for t in self._tokens) / max(len(self._tokens), 1)
        df: Counter = Counter()
        for t in self._tokens:
            df.update(set(t))
        n = len(self.passages)
        self._idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def _score(self, query_tokens: list[str], i: int) -> float:
        tf, length = self._tf[i], len(self._tokens[i])
        score = 0.0
        for w in query_tokens:
            if w not in tf:
                continue
            f = tf[w]
            denom = f + self.k1 * (1 - self.b + self.b * length / (self._avg_len or 1))
            score += self._idf[w] * f * (self.k1 + 1) / denom
        return score

    def search(self, query: str, k: int = 3) -> list[Passage]:
        q = tokenize(query)
        scored = [(self._score(q, i), i) for i in range(len(self.passages))]
        scored.sort(key=lambda x: -x[0])
        return [
            Passage(self.passages[i].source, self.passages[i].text, round(s, 4))
            for s, i in scored[:k]
            if s > 0
        ]


def retrieval_recall(cases: list[TestCase], retriever: Retriever, k: int = 3) -> float:
    """Fraction of gold evidence sentences found in the top-k, over answerable cases."""
    found = total = 0
    for c in cases:
        if not c.answerable:
            continue
        retrieved = {p.text for p in retriever.search(c.question, k)}
        for e in c.evidence:
            total += 1
            found += e.quote in retrieved
    return found / total if total else 0.0
