"""Real LLM clients. Pick one with make_client(); all satisfy the LLMClient protocol."""
from __future__ import annotations

import json
import os
import time
import urllib.request

from .llm import LLMClient, LLMResponse, MockClient


class AnthropicClient:
    """Needs `pip install anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: str, max_tokens: int = 512):
        import anthropic  # imported lazily so the rest works without it

        self._client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        kwargs = dict(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        start = time.perf_counter()
        msg = self._client.messages.create(**kwargs)
        latency = time.perf_counter() - start
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResponse(text, latency, msg.usage.input_tokens, msg.usage.output_tokens)


class OpenAICompatClient:
    """Any OpenAI-style /chat/completions API (OpenAI, Ollama, vLLM, LM Studio...).

    Env: OPENAI_BASE_URL (default https://api.openai.com/v1), OPENAI_API_KEY (optional for local).
    Ollama example: OPENAI_BASE_URL=http://localhost:11434/v1
    """

    def __init__(self, model: str, max_tokens: int = 512):
        self.model, self.max_tokens = model, max_tokens
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body = json.dumps(
            {"model": self.model, "messages": messages, "max_tokens": self.max_tokens, "temperature": 0}
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", body, headers)
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
        latency = time.perf_counter() - start
        usage = data.get("usage", {})
        return LLMResponse(
            data["choices"][0]["message"]["content"], latency,
            usage.get("prompt_tokens"), usage.get("completion_tokens"),
        )


def make_client(provider: str, model: str | None = None) -> LLMClient:
    if provider == "mock":
        return MockClient()
    if not model:
        raise ValueError("--model is required for real providers")
    if provider == "anthropic":
        return AnthropicClient(model)
    if provider == "openai-compat":
        return OpenAICompatClient(model)
    raise ValueError(f"unknown provider: {provider}")
