"""Provider-agnostic LLM interface.

The real hosted/local client is added in Phase 2 once a model is chosen.
Everything else depends only on `LLMClient`, so the provider can be swapped.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMClient(Protocol):
    def complete(self, prompt: str, system: str | None = None) -> LLMResponse: ...


class MockClient:
    """Deterministic client for tests and offline development."""

    def __init__(self, responder: Callable[[str], str] | str = "mock answer"):
        self._responder = responder

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        start = time.perf_counter()
        text = self._responder(prompt) if callable(self._responder) else self._responder
        return LLMResponse(text=text, latency_s=time.perf_counter() - start)
