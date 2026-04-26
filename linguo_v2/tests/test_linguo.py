"""
tests/test_linguo.py — Unit and integration tests

Run with:
    pytest tests/ -v

Note: agent tests mock the OpenAI-compatible client used to talk to Ollama,
so no local Ollama instance is required to run the test suite.
"""

import pytest
from unittest.mock import patch, MagicMock

from state.models import UserState, WordRecord, GeneratedSentence, EvaluationResult
from rag.dictionary import RAGDictionary, DictionaryEntry
from config import MASTERY_CORRECT_THRESHOLD


# ── UserState ──────────────────────────────────────────────────────────────────

class TestUserState:
    def test_initial_state(self):
        state = UserState()
        assert state.total_seen == 0
        assert state.streak == 0
        assert state.mastered_count == 0
        assert state.level == "beginner"

    def test_record_word(self):
        state = UserState()
        state.record_word("gato", "cat", "Spanish")
        assert "gato" in state.vocab
        assert state.total_seen == 1

    def test_no_duplicate_words(self):
        state = UserState()
        state.record_word("gato", "cat", "Spanish")
        state.record_word("gato", "cat", "Spanish")
        assert state.total_seen == 1

    def test_streak_increments_on_correct(self):
        state = UserState()
        state.record_word("gato", "cat", "Spanish")
        state.record_answer("gato", correct=True)
        assert state.streak == 1

    def test_streak_resets_on_wrong(self):
        state = UserState()
        state.record_word("gato", "cat", "Spanish")
        state.record_answer("gato", correct=True)
        state.record_answer("gato", correct=False)
        assert state.streak == 0

    def test_mastery_threshold(self):
        state = UserState()
        state.record_word("gato", "cat", "Spanish")
        for _ in range(MASTERY_CORRECT_THRESHOLD):
            state.record_answer("gato", correct=True)
        assert state.vocab["gato"].mastered is True
        assert state.mastered_count == 1

    def test_level_progression(self):
        state = UserState()
        for i in range(5):
            word = f"word{i}"
            state.record_word(word, f"meaning{i}", "Spanish")
            for _ in range(MASTERY_CORRECT_THRESHOLD):
                state.record_answer(word, correct=True)
        assert state.level == "intermediate"


# ── WordRecord ─────────────────────────────────────────────────────────────────

class TestWordRecord:
    def test_accuracy_zero_attempts(self):
        rec = WordRecord(word="hola", meaning="hello", lang="Spanish")
        assert rec.accuracy == 0.0

    def test_accuracy_calculation(self):
        rec = WordRecord(word="hola", meaning="hello", lang="Spanish", correct=3, attempts=4)
        assert rec.accuracy == 0.75

    def test_mastered_false_by_default(self):
        rec = WordRecord(word="hola", meaning="hello", lang="Spanish")
        assert rec.mastered is False


# ── RAGDictionary ──────────────────────────────────────────────────────────────

class TestRAGDictionary:
    def _make_entry(self, word="gato", lang="Spanish", meaning="cat"):
        return DictionaryEntry(
            foreign_word=word,
            language=lang,
            english_meaning=meaning,
            part_of_speech="noun",
            example_context=f"El {word} es pequeño.",
        )

    def test_add_and_size(self):
        rag = RAGDictionary()
        rag.add_entry(self._make_entry())
        assert rag.size() == 1

    def test_no_duplicates(self):
        rag = RAGDictionary()
        rag.add_entry(self._make_entry())
        rag.add_entry(self._make_entry())  # duplicate
        assert rag.size() == 1

    def test_exact_lookup_found(self):
        rag = RAGDictionary()
        rag.add_entry(self._make_entry("perro", "Spanish", "dog"))
        result = rag.exact_lookup("perro", "Spanish")
        assert result is not None
        assert result.english_meaning == "dog"

    def test_exact_lookup_not_found(self):
        rag = RAGDictionary()
        result = rag.exact_lookup("xyz123", "Spanish")
        assert result is None

    def test_exact_lookup_case_insensitive(self):
        rag = RAGDictionary()
        rag.add_entry(self._make_entry("Perro", "Spanish", "dog"))
        result = rag.exact_lookup("perro", "Spanish")
        assert result is not None

    def test_export_import_json(self, tmp_path):
        rag = RAGDictionary()
        rag.add_entry(self._make_entry("gato", "Spanish", "cat"))
        path = str(tmp_path / "dict.json")
        rag.export_json(path)

        rag2 = RAGDictionary()
        rag2.import_json(path)
        assert rag2.size() == 1
        assert rag2.exact_lookup("gato", "Spanish") is not None


# ── GeneratedSentence model ────────────────────────────────────────────────────

class TestGeneratedSentence:
    def test_valid_construction(self):
        s = GeneratedSentence(
            sentence="The gato sat on the mat.",
            foreign_word="gato",
            english_meaning="cat",
            part_of_speech="noun",
            example_context="El gato es bonito.",
            difficulty="easy",
        )
        assert s.foreign_word == "gato"

    def test_missing_field_raises(self):
        with pytest.raises(Exception):
            GeneratedSentence(sentence="Hello")  # type: ignore


# ── Orchestrator (mocked Ollama calls) ────────────────────────────────────────

class TestOrchestrator:
    """Integration tests using mocked Ollama/OpenAI-compatible API responses."""

    def _mock_completion(self, content: str):
        """Build a mock openai ChatCompletion response."""
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        return response

    def _sentence_json(self):
        return (
            '{"sentence": "The chat sat on the mat.", '
            '"foreign_word": "chat", '
            '"english_meaning": "cat", '
            '"part_of_speech": "noun", '
            '"example_context": "Le chat est mignon.", '
            '"difficulty": "easy"}'
        )

    def _eval_json(self, correct=True):
        return (
            f'{{"correct": {str(correct).lower()}, '
            f'"feedback": "Well done!", '
            f'"score": {90 if correct else 20}}}'
        )

    @patch("agents.base.OpenAI")
    def test_generate_sentence(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_completion(self._sentence_json())

        from agents.orchestrator import Orchestrator
        orch = Orchestrator()
        sentence, logs = orch.generate_sentence("French", "everyday life")
        assert sentence.foreign_word == "chat"
        assert "chat" in orch.user_state.vocab
        assert any("sentence-agent" in log for log in logs)

    @patch("agents.base.OpenAI")
    def test_check_correct_answer(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            self._mock_completion(self._sentence_json()),
            self._mock_completion(self._eval_json(correct=True)),
        ]

        from agents.orchestrator import Orchestrator
        orch = Orchestrator()
        orch.generate_sentence("French", "everyday life")
        result, logs = orch.check_answer("cat")
        assert result.correct is True
        assert orch.user_state.streak == 1

    @patch("agents.base.OpenAI")
    def test_check_wrong_answer_resets_streak(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            self._mock_completion(self._sentence_json()),
            self._mock_completion(self._eval_json(correct=True)),
            self._mock_completion(self._sentence_json()),
            self._mock_completion(self._eval_json(correct=False)),
        ]

        from agents.orchestrator import Orchestrator
        orch = Orchestrator()
        orch.generate_sentence("French", "everyday life")
        orch.check_answer("cat")
        orch.generate_sentence("French", "everyday life")
        result, _ = orch.check_answer("wrong")
        assert result.correct is False
        assert orch.user_state.streak == 0

    @patch("agents.base.OpenAI")
    def test_reset_clears_state(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_completion(self._sentence_json())

        from agents.orchestrator import Orchestrator
        orch = Orchestrator()
        orch.generate_sentence("French", "everyday life")
        orch.reset_session()
        assert orch.user_state.total_seen == 0
        assert orch.user_state.streak == 0
