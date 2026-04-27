"""
agents/base.py — Abstract base class for all agents

Each agent:
  - Receives a structured prompt context
  - Calls the local Ollama API (OpenAI-compatible endpoint)
  - Parses and validates the structured JSON response
  - Emits log messages for the orchestrator to collect

Ollama must be running locally:
    https://ollama.com  ->  ollama pull llama3.2
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Generator

from openai import OpenAI

from config import OLLAMA_BASE_URL, MODEL


class BaseAgent(ABC):
    """Shared plumbing for all Linguo agents."""

    name: str = "base-agent"

    def __init__(self):
        # Ollama exposes an OpenAI-compatible API at /v1 — no key needed.
        self.client = OpenAI(
            base_url=f"{OLLAMA_BASE_URL}/v1",
            api_key="ollama",          # required by the openai SDK, value ignored by Ollama
        )
        self.logs: list[str] = []

    def log(self, msg: str) -> None:
        entry = f"[{self.name}] {msg}"
        self.logs.append(entry)

    def clear_logs(self) -> None:
        self.logs.clear()

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def _call(self, prompt: str, max_tokens: int = 1000) -> str:
        """Single-turn call to Ollama; returns raw text content."""
        response = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def _call_streaming(self, prompt: str, max_tokens: int = 1000) -> Generator[str, None, None]:
        """Streaming call to Ollama; yields text chunks as they arrive."""
        stream = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _parse_json(self, raw: str) -> dict[str, Any]:
        """
        Robustly parse JSON from LLM output, handling common Ollama quirks:
          - Markdown fences (```json ... ```)
          - Bold/italic markers around words (**word** or *word*)
          - Truncated output missing the closing brace
          - Leading/trailing prose outside the JSON object
        """
        # 1. Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

        # 2. Strip bold/italic markdown inside string values (**text** or *text*)
        cleaned = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", cleaned)

        # 3. Extract just the first complete {...} block, ignoring prose before/after
        start = cleaned.find("{")
        end   = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        elif start != -1:
            # Truncated — model stopped before closing brace; attempt to close it
            cleaned = cleaned[start:].rstrip(",\n\r ") + "\n}"

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"JSON parse error in {self.name}: {e}\nRaw: {raw[:400]}"
            )

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the agent's task and return a typed result."""
        ...
