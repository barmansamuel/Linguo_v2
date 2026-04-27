"""
agents/orchestrator.py — Central coordinator for the multi-agent workflow

Persistence layer (new):
  SQLiteStore  — durable cross-session storage (vocab, RAG entries, history, stats)
  MemoryStore  — in-session knowledge graph (agent working context, recent words,
                 struggled words, topic coverage)

On startup:
  1. SQLiteStore loads vocab, stats, and RAG entries from disk → restores UserState
     and pre-populates the RAG dictionary so the agent has context immediately.

On every turn:
  generate_sentence()
    ├─ MemoryStore.record_topic_covered()
    ├─ SentenceAgent.run()  (reads MemoryStore for struggled/recent words)
    ├─ SQLiteStore.save_word() + save_rag_entry()
    └─ MemoryStore.record_word_seen() + record_sentence()

  check_answer()
    ├─ EvaluatorAgent.run()
    ├─ UserState.record_answer()
    ├─ SQLiteStore.save_word() + append_history() + save_stats()
    └─ MemoryStore.record_answer()

  get_progress()
    └─ ProgressAgent.run()  (receives MemoryStore.get_session_summary())

On reset_session():
  SQLiteStore persists — vocab survives.
  MemoryStore.clear() — in-session context resets.
  UserState reloads from SQLite so mastery carries over.
"""

from __future__ import annotations

from rag.dictionary import RAGDictionary, DictionaryEntry
from state.models import UserState, WordRecord, GeneratedSentence, EvaluationResult
from agents.sentence_agent import SentenceAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.hint_agent import HintAgent
from agents.progress_agent import ProgressAgent
from persistence.sqlite_store import SQLiteStore
from persistence.memory_store import MemoryStore


class Orchestrator:
    def __init__(self):
        # ── Persistence ────────────────────────────────────────────────────────
        self.db     = SQLiteStore()
        self.memory = MemoryStore()

        # ── RAG + state: restored from SQLite ─────────────────────────────────
        self.rag   = RAGDictionary()
        self.state = self._load_state_from_db()

        # ── Agents ────────────────────────────────────────────────────────────
        self.sentence_agent  = SentenceAgent(self.rag, self.memory)
        self.evaluator_agent = EvaluatorAgent(self.rag)
        self.hint_agent      = HintAgent(self.rag)
        self.progress_agent  = ProgressAgent()

        self._current: GeneratedSentence | None = None
        self._current_lang: str = "Spanish"

    # ── Startup restore ────────────────────────────────────────────────────────

    def _load_state_from_db(self) -> UserState:
        """Restore UserState and RAG dictionary from SQLite on startup."""
        state = UserState()

        # Restore vocab
        for row in self.db.load_all_vocab():
            rec = WordRecord(
                word=row["word"],
                meaning=row["meaning"],
                lang=row["lang"],
                correct=row["correct"],
                attempts=row["attempts"],
            )
            state.vocab[row["word"]] = rec
            if rec.attempts > 0:
                state.total_seen += 1

        # Restore stats (streak, total_seen override the counted value)
        stats = self.db.load_stats()
        state.streak     = stats["streak"]
        state.total_seen = stats["total_seen"]

        # Restore RAG dictionary entries
        for row in self.db.load_rag_entries():
            self.rag.add_entry(DictionaryEntry(
                foreign_word=row["foreign_word"],
                language=row["language"],
                english_meaning=row["english_meaning"],
                part_of_speech=row["part_of_speech"],
                example_context=row["example_context"],
            ))  # romanization stored in DB but not needed in RAG vector index

        return state

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_sentence(self, language: str, topic: str) -> tuple[GeneratedSentence, list[str]]:
        self._current_lang = language
        logs = [
            f"[orchestrator] routing to sentence-agent "
            f"(lang={language}, topic={topic}, level={self.state.level})",
            f"[memory] recording topic: {topic}",
        ]

        # Memory: track topic coverage
        self.memory.record_topic_covered(topic)

        result = self.sentence_agent.run(
            language=language,
            topic=topic,
            user_state=self.state,
        )
        logs += self.sentence_agent.logs

        # UserState
        self.state.record_word(result.foreign_word, result.english_meaning, language)

        # SQLite: persist word + RAG entry
        rec = self.state.vocab[result.foreign_word]
        self.db.save_word(result.foreign_word, result.english_meaning, language,
                          rec.correct, rec.attempts)
        self.db.save_rag_entry(result.foreign_word, language, result.english_meaning,
                               result.part_of_speech, result.example_context,
                               getattr(result, 'romanization', ''))
        logs.append(f"[sqlite] saved word '{result.foreign_word}' to db")

        # Memory: record word + sentence context
        self.memory.record_word_seen(result.foreign_word, language,
                                     result.english_meaning, result.sentence)
        self.memory.record_sentence(result.sentence, result.foreign_word)
        logs.append(f"[memory] recorded word + sentence context")

        self._current = result
        logs.append("[orchestrator] ready for user input")
        return result, logs

    def check_answer(self, guess: str) -> tuple[EvaluationResult, list[str]]:
        if self._current is None:
            raise RuntimeError("No active sentence — call generate_sentence() first.")

        logs = ["[orchestrator] routing to evaluator-agent"]

        result = self.evaluator_agent.run(
            language=self._current_lang,
            foreign_word=self._current.foreign_word,
            correct_meaning=self._current.english_meaning,
            guess=guess,
        )
        logs += self.evaluator_agent.logs

        # UserState
        self.state.record_answer(self._current.foreign_word, result.correct)

        # SQLite: persist updated word stats + history + session stats
        rec = self.state.vocab[self._current.foreign_word]
        self.db.save_word(self._current.foreign_word, self._current.english_meaning,
                          self._current_lang, rec.correct, rec.attempts)
        self.db.append_history(self._current.foreign_word, result.correct)
        self.db.save_stats(self.state.streak, self.state.total_seen)
        logs.append(f"[sqlite] persisted answer + stats (streak={self.state.streak})")

        # Memory: record outcome
        self.memory.record_answer(self._current.foreign_word, self._current_lang,
                                  result.correct, guess)
        logs.append(
            f"[memory] recorded {'correct' if result.correct else 'incorrect'} answer"
        )

        logs.append(
            f"[orchestrator] state updated: streak={self.state.streak}, "
            f"mastered={self.state.mastered_count}"
        )
        return result, logs

    def get_hint(self) -> tuple[str, list[str]]:
        if self._current is None:
            raise RuntimeError("No active sentence.")
        logs = ["[orchestrator] routing to hint-agent"]
        hint = self.hint_agent.run(
            language=self._current_lang,
            foreign_word=self._current.foreign_word,
            correct_meaning=self._current.english_meaning,
            sentence=self._current.sentence,
        )
        logs += self.hint_agent.logs
        return hint, logs

    def get_progress(self) -> tuple[dict, list[str]]:
        logs = ["[orchestrator] routing to progress-agent"]
        # Pass memory summary so ProgressAgent has richer session context
        mem_summary = self.memory.get_session_summary()
        logs.append(f"[memory] session summary: {len(mem_summary['struggled_words'])} struggled, "
                    f"{len(mem_summary['topics_covered'])} topics covered")
        analysis = self.progress_agent.run(self.state, mem_summary)
        logs += self.progress_agent.logs
        return analysis, logs

    # ── Convenience accessors ──────────────────────────────────────────────────

    @property
    def user_state(self) -> UserState:
        return self.state

    @property
    def current_sentence(self) -> GeneratedSentence | None:
        return self._current

    def reset_session(self) -> None:
        """
        Reset in-session context but keep all durable progress.
        Vocab and mastery survive — the memory graph and current sentence reset.
        """
        # Memory graph resets (in-session context only)
        self.memory.clear()
        self._current = None
        # Reload state from SQLite so mastery/vocab carry over
        self.state = self._load_state_from_db()
