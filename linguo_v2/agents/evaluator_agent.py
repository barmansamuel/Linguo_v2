"""
agents/evaluator_agent.py — Grades user guesses with semantic flexibility

Responsibilities:
  1. Accept the foreign word, its correct meaning, and the user's guess
  2. Use the RAG dictionary to find any synonymous accepted meanings
  3. Prompt the LLM to evaluate correctness (synonyms, slight misspellings OK)
  4. Return a validated EvaluationResult
"""

from __future__ import annotations

from rag.dictionary import RAGDictionary
from state.models import EvaluationResult
from agents.base import BaseAgent


EVALUATOR_PROMPT = """You are a language learning evaluator inside a multi-agent system.

A student is learning {language} and guessed the meaning of a foreign word.

Foreign word:    "{foreign_word}"
Correct meaning: "{correct_meaning}"
Student's guess: "{guess}"
RAG synonyms:    {rag_synonyms}

Is the guess correct or acceptable?
Criteria:
- Accept synonyms and near-synonyms
- Accept minor misspellings
- Accept partial matches if they capture the core meaning
- Reject guesses that are completely wrong or too vague

Return ONLY valid JSON:
{{
  "correct":  true | false,
  "feedback": "<one encouraging sentence: confirm if right, or gently correct and give the right answer>",
  "score":    <integer 0–100 reflecting how close the guess was>
}}"""


class EvaluatorAgent(BaseAgent):
    name = "evaluator-agent"

    def __init__(self, rag: RAGDictionary):
        super().__init__()
        self.rag = rag

    def run(
        self,
        language: str,
        foreign_word: str,
        correct_meaning: str,
        guess: str,
    ) -> EvaluationResult:
        self.clear_logs()
        self.log(f"Evaluating guess='{guess}' for word='{foreign_word}' ({language})")

        # RAG: pull any synonymous accepted meanings for this word
        rag_results = self.rag.lookup(foreign_word, language, top_k=3)
        rag_synonyms = (
            ", ".join(e.english_meaning for e in rag_results)
            or "none"
        )
        self.log(f"RAG synonyms found: {rag_synonyms}")

        prompt = EVALUATOR_PROMPT.format(
            language=language,
            foreign_word=foreign_word,
            correct_meaning=correct_meaning,
            guess=guess,
            rag_synonyms=rag_synonyms,
        )

        raw = self._call(prompt, max_tokens=300)
        data = self._parse_json(raw)
        result = EvaluationResult(**data)

        self.log(f"Result: correct={result.correct}, score={result.score}")
        return result
