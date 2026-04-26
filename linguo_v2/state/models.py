"""
state/models.py — Pydantic models for all data structures
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class Difficulty(str, Enum):
    easy   = "easy"
    medium = "medium"
    hard   = "hard"


class PartOfSpeech(str, Enum):
    noun      = "noun"
    verb      = "verb"
    adjective = "adjective"
    adverb    = "adverb"
    phrase    = "phrase"
    other     = "other"


class GeneratedSentence(BaseModel):
    """Output from the Sentence Agent."""
    sentence:        str
    foreign_word:    str
    english_meaning: str
    part_of_speech:  str
    example_context: str
    difficulty:      str


class EvaluationResult(BaseModel):
    """Output from the Evaluator Agent."""
    correct:  bool
    feedback: str
    score:    int = Field(ge=0, le=100)


class WordRecord(BaseModel):
    """Per-word progress record stored in UserState."""
    word:    str
    meaning: str
    lang:    str
    correct: int = 0
    attempts: int = 0

    @property
    def mastered(self) -> bool:
        from config import MASTERY_CORRECT_THRESHOLD
        return self.correct >= MASTERY_CORRECT_THRESHOLD

    @property
    def accuracy(self) -> float:
        if self.attempts == 0:
            return 0.0
        return round(self.correct / self.attempts, 2)


class UserState(BaseModel):
    """Full session state for one user."""
    vocab:      dict[str, WordRecord] = Field(default_factory=dict)
    streak:     int = 0
    total_seen: int = 0
    history:    list[dict] = Field(default_factory=list)

    # ── Computed helpers ───────────────────────────────────────────────────────
    @property
    def mastered_count(self) -> int:
        return sum(1 for w in self.vocab.values() if w.mastered)

    @property
    def level(self) -> str:
        from config import LEVEL_THRESHOLDS
        n = self.mastered_count
        for lvl, (lo, hi) in LEVEL_THRESHOLDS.items():
            if lo <= n <= hi:
                return lvl
        return "advanced"

    @property
    def mastered_words(self) -> list[str]:
        return [w for w, rec in self.vocab.items() if rec.mastered]

    def record_word(self, word: str, meaning: str, lang: str) -> None:
        """Register a new word if not already tracked."""
        if word not in self.vocab:
            self.vocab[word] = WordRecord(word=word, meaning=meaning, lang=lang)
            self.total_seen += 1

    def record_answer(self, word: str, correct: bool) -> None:
        """Update stats after the user submits an answer."""
        rec = self.vocab.get(word)
        if rec is None:
            return
        rec.attempts += 1
        if correct:
            rec.correct += 1
            self.streak += 1
        else:
            self.streak = 0
        self.history.append({"word": word, "correct": correct})
