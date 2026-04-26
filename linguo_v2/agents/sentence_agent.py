"""
agents/sentence_agent.py — Generates contextual sentences with an embedded foreign word

Responsibilities:
  1. Receive user level, language, topic, and vocabulary history
  2. Query the RAG dictionary for existing words to avoid repetition
  3. Prompt the LLM to produce a new sentence with one embedded foreign word
  4. Return a validated GeneratedSentence model
  5. Register the new word into the RAG dictionary
"""

from __future__ import annotations

import re

from rag.dictionary import RAGDictionary, DictionaryEntry
from state.models import GeneratedSentence, UserState
from agents.base import BaseAgent
from persistence.memory_store import MemoryStore


SENTENCE_PROMPT = """You are a language learning sentence generator. Your entire job is to embed ONE real {language} word inside an English sentence so a learner can guess its meaning from context.

TASK
----
Write one English sentence about "{topic}" (difficulty: {level}).
Replace exactly ONE English word in that sentence with its genuine {language} translation.

CRITICAL REQUIREMENTS — read carefully
---------------------------------------
1. The foreign word MUST be a real {language} word, written in the {language} script/alphabet.
   - Spanish example  → "perro" (not "dog")
   - French example   → "maison" (not "house")
   - Japanese example → "猫" or "ねこ" (not "cat")
   - Mandarin example → "书" (not "book")
   - Korean example   → "물" (not "water")
   - German example   → "Hund" (not "dog")

2. The word must NOT be a loanword that is identical or nearly identical in English
   (e.g. do NOT use "sandwich", "taxi", "hotel", "pizza", "café" — these are the same in both languages and give nothing to learn).

3. The word must appear INSIDE the sentence field, replacing the English word.

4. english_meaning must be the English translation of the foreign word you chose.

5. Do NOT reuse: {avoid_words}

6. Return ONLY a JSON object — no markdown fences, no explanation, no extra text.

LEVEL GUIDANCE
--------------
beginner     → very common concrete nouns or verbs (colors, food items, family members, basic actions)
intermediate → everyday vocabulary, emotions, common verbs
advanced     → richer vocabulary, abstract nouns, idiomatic phrases

EXAMPLE OUTPUT (Spanish, topic: food)
--------------------------------------
{{"sentence": "She put the leche in her coffee.", "foreign_word": "leche", "english_meaning": "milk", "part_of_speech": "noun", "example_context": "La leche está fría.", "difficulty": "easy"}}

NOW GENERATE (language: {language}, topic: {topic}, level: {level})
Do NOT use **bold**, *italic*, or any markdown inside the JSON values.
JSON:"""


# Common loanwords that are the same (or near-identical) across many languages —
# if the model picks one of these as the "foreign" word, we retry.
_LOANWORDS = {
    "sandwich", "taxi", "hotel", "pizza", "café", "cafe", "internet",
    "email", "bus", "radio", "television", "chocolate", "tomato",
    "banana", "mango", "sofa", "pasta", "sushi", "yoga", "karate",
    "sport", "football", "basketball", "tennis", "golf", "bar",
    "menu", "restaurant", "supermarket", "airport", "metro",
}


class SentenceAgent(BaseAgent):
    name = "sentence-agent"

    def __init__(self, rag: RAGDictionary, memory: MemoryStore | None = None):
        super().__init__()
        self.rag    = rag
        self.memory = memory

    def run(
        self,
        language: str,
        topic: str,
        user_state: UserState,
        max_retries: int = 3,
    ) -> GeneratedSentence:
        self.clear_logs()
        level = user_state.level
        mastered = user_state.mastered_words[-10:]

        rag_results = self.rag.lookup(topic, language, top_k=5)
        rag_context = (
            ", ".join(f"{e.foreign_word}={e.english_meaning}" for e in rag_results)
            or "none yet"
        )

        # Memory: pull struggled + recent words for smarter avoidance
        struggled_words = []
        recent_words    = []
        if self.memory:
            struggled_words = self.memory.get_struggled_words()
            recent_words    = self.memory.get_recent_words(5)
            self.log(f"[memory] {len(struggled_words)} struggled, {len(recent_words)} recent words")

        # Combine all words to avoid: mastered + struggled this session + recent
        avoid_set = list(dict.fromkeys(mastered + struggled_words + recent_words))

        self.log(f"level={level}, topic={topic}, lang={language}")
        self.log(f"RAG context ({len(rag_results)} entries): {rag_context}")
        self.log(f"Avoiding {len(avoid_set)} words (mastered + struggled + recent)")

        prompt = SENTENCE_PROMPT.format(
            topic=topic,
            level=level,
            language=language,
            avoid_words=", ".join(avoid_set) or "none",
            rag_context=rag_context,
        )

        last_error = None
        for attempt in range(1, max_retries + 1):
            self.log(f"LLM call attempt {attempt}/{max_retries}...")
            try:
                raw = self._call(prompt, max_tokens=800)
                data = self._parse_json(raw)

                # Clean any residual markdown bold/italic in field values
                for field in ("foreign_word", "english_meaning", "sentence"):
                    if field in data and isinstance(data[field], str):
                        data[field] = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", data[field]).strip()

                result = GeneratedSentence(**data)

                # Validate: foreign word must actually appear in the sentence
                if result.foreign_word not in result.sentence:
                    raise ValueError(
                        f"foreign_word '{result.foreign_word}' not found in sentence: '{result.sentence}'"
                    )

                # Validate: foreign word must not be a known loanword
                if result.foreign_word.lower() in _LOANWORDS:
                    raise ValueError(
                        f"'{result.foreign_word}' is a loanword — same in English, not useful to learn"
                    )

                # Validate: foreign word should not be identical to its English meaning
                if result.foreign_word.lower() == result.english_meaning.lower():
                    raise ValueError(
                        f"foreign_word '{result.foreign_word}' is identical to english_meaning — not a real translation"
                    )

                # All checks passed
                self.rag.add_entry(DictionaryEntry(
                    foreign_word=result.foreign_word,
                    language=language,
                    english_meaning=result.english_meaning,
                    part_of_speech=result.part_of_speech,
                    example_context=result.example_context,
                ))
                self.log(f"✓ '{result.foreign_word}' → '{result.english_meaning}' (RAG size={self.rag.size()})")
                return result

            except Exception as e:
                last_error = e
                self.log(f"Attempt {attempt} failed: {e} — retrying...")

        raise RuntimeError(
            f"SentenceAgent failed after {max_retries} attempts. Last error: {last_error}"
        )
