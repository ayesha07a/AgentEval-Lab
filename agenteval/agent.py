"""Baseline RAG agent: retrieve top-k sentences, then ask the LLM.

The prompt is intentionally plain. It does NOT tell the model to abstain when the
context lacks the answer; that is a Phase 6 improvement to be measured against this.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .llm import LLMClient
from .retrieval import Passage, Retriever

PROMPT_VERSION = "baseline-v1"
SYSTEM = "You answer questions using the provided context."


@dataclass
class AgentResult:
    question: str
    answer: str
    passages: list[Passage] = field(default_factory=list)
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


def build_prompt(question: str, passages: list[Passage]) -> str:
    context = "\n".join(f"[{i}] ({p.source}) {p.text}" for i, p in enumerate(passages, 1))
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely."


class Agent:
    def __init__(self, client: LLMClient, retriever: Retriever, k: int = 3):
        self.client, self.retriever, self.k = client, retriever, k

    def answer(self, question: str) -> AgentResult:
        passages = self.retriever.search(question, self.k)
        try:
            r = self.client.complete(build_prompt(question, passages), system=SYSTEM)
        except Exception as e:  # record failures instead of crashing the whole run
            return AgentResult(question, "", passages, error=f"{type(e).__name__}: {e}")
        return AgentResult(
            question, r.text.strip(), passages, r.latency_s, r.input_tokens, r.output_tokens
        )
