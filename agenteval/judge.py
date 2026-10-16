"""Optional LLM judge. Returns None whenever the verdict can't be parsed, so the
caller can fall back to the deterministic heuristics instead of guessing."""
from __future__ import annotations

import json
import re

from .llm import LLMClient

_JSON = re.compile(r"\{.*?\}", re.DOTALL)

CORRECT_PROMPT = """You are grading an answer against a reference answer.
Question: {question}
Reference answer: {expected}
Candidate answer: {answer}

Is the candidate answer correct, i.e. does it state the same facts as the reference
and contradict nothing in it? Reply with JSON only: {{"correct": true}} or {{"correct": false}}."""

GROUNDED_PROMPT = """Context passages:
{context}

Answer to check: {answer}

Is every factual claim in the answer supported by the context passages (derived values such as
a computed difference count as supported)? Reply with JSON only:
{{"grounded": true}} or {{"grounded": false}}."""


def _parse(text: str, key: str) -> bool | None:
    m = _JSON.search(text)
    if not m:
        return None
    try:
        val = json.loads(m.group(0)).get(key)
    except (json.JSONDecodeError, AttributeError):
        return None
    return val if isinstance(val, bool) else None


class LLMJudge:
    def __init__(self, client: LLMClient):
        self.client = client

    def _ask(self, prompt: str, key: str) -> bool | None:
        try:
            return _parse(self.client.complete(prompt).text, key)
        except Exception:
            return None

    def correct(self, question: str, expected: str, answer: str) -> bool | None:
        return self._ask(CORRECT_PROMPT.format(question=question, expected=expected, answer=answer), "correct")

    def grounded(self, answer: str, passages: list[str]) -> bool | None:
        ctx = "\n".join(f"- {p}" for p in passages)
        return self._ask(GROUNDED_PROMPT.format(context=ctx, answer=answer), "grounded")
