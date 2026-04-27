"""
agents/hint_agent.py — Generates contextual hints without revealing the answer

Responsibilities:
  1. Accept the sentence, the foreign word, and its correct meaning
  2. Query the RAG dictionary for etymology or usage context
  3. Generate a subtle, helpful hint that doesn't give away the answer
"""

from __future__ import annotations

from rag.dictionary import RAGDictionary
from agents.base import BaseAgent


HINT_PROMPT = """You are a hint generator inside a language learning multi-agent system.

A student is trying to guess the meaning of "{foreign_word}" ({language}) in this sentence:
"{sentence}"

The correct meaning is "{correct_meaning}" — but do NOT reveal it directly.
RAG context for this word: {rag_context}

Generate ONE helpful hint (max 20 words) that:
- Uses context clues from the sentence
- References the word's part of speech if helpful
- Does NOT give away the answer outright

Return ONLY the hint text, no JSON, no quotes, no preamble."""


class HintAgent(BaseAgent):
    name = "hint-agent"

    def __init__(self, rag: RAGDictionary):
        super().__init__()
        self.rag = rag

    def run(
        self,
        language: str,
        foreign_word: str,
        correct_meaning: str,
        sentence: str,
    ) -> str:
        self.clear_logs()
        self.log(f"Generating hint for '{foreign_word}' ({language})")

        entry = self.rag.exact_lookup(foreign_word, language)
        rag_context = (
            f"part of speech: {entry.part_of_speech}; example: {entry.example_context}"
            if entry
            else "no additional context"
        )
        self.log(f"RAG context: {rag_context}")

        prompt = HINT_PROMPT.format(
            language=language,
            foreign_word=foreign_word,
            correct_meaning=correct_meaning,
            sentence=sentence,
            rag_context=rag_context,
        )

        hint = self._call(prompt, max_tokens=100).strip()
        self.log(f"Hint generated: {hint[:60]}...")
        return hint
