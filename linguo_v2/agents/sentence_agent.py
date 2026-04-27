"""
agents/sentence_agent.py — Generates contextual sentences with an embedded foreign word

New vocab-first architecture:
  1. VocabLoader.pick_word() selects a verified word from the curated JSON dictionary
     → the word, romanization, and English meaning are all known before the LLM is called
  2. The LLM is asked only to write ONE English sentence using that specific word
     → eliminates all CJK romanization errors (word comes from JSON, not the model)
  3. Self-repair and romanization fallback are kept as a safety net for edge cases
  4. RAG context is still used to avoid repeating recently seen words
"""

from __future__ import annotations

import re

from rag.dictionary import RAGDictionary, DictionaryEntry
from state.models import GeneratedSentence, UserState
from agents.base import BaseAgent
from persistence.memory_store import MemoryStore
from data.vocab.vocab_loader import VocabLoader, VocabEntry


# ── Language metadata ──────────────────────────────────────────────────────────

CJK_LANGUAGES = {"Mandarin", "Japanese", "Korean"}

CJK_SCRIPT_NAME = {
    "Mandarin": "Chinese characters (汉字)",
    "Japanese": "Kanji or Kana (漢字・かな)",
    "Korean":   "Hangul (한글)",
}

ROMANIZATION_PROMPT = """Give the {label} pronunciation for the {language} word "{word}".
Return only the romanization string — no punctuation, no explanation, nothing else."""

ROMANIZATION_LABEL = {
    "Mandarin": "Pinyin",
    "Japanese": "Romaji",
    "Korean":   "Romanization",
}


# ── Prompts ────────────────────────────────────────────────────────────────────

SENTENCE_PROMPT = """You write language learning sentences. Output ONE JSON object only.

TASK
----
Write an English sentence about "{topic}" that naturally uses the {english_meaning} word below.
Replace the English word with the actual {foreign_word} word in {language} — do NOT use the English meaning in the sentence. The sentence should be natural and contextually relevant to the topic.
The word is already chosen — do NOT change it or use a different word.

GIVEN WORD
----------
  {language} word : {foreign_word}
  Pronunciation   : {romanization_hint}
  English meaning : {english_meaning}
  Part of speech  : {pos}

FIELD RULES
-----------
  "sentence"        → English sentence containing "{foreign_word}" verbatim (not romanization)
  "foreign_word"    → Copy exactly: {foreign_word}
  "romanization"    → Copy exactly: {romanization_hint}
  "english_meaning" → Copy exactly: {english_meaning}
  "part_of_speech"  → Copy exactly: {pos}
  "example_context" → One short sentence in {language} using the word
  "difficulty"      → {difficulty}

RULES
-----
- The word "{foreign_word}" must appear literally in the sentence — not its romanization or English meaning
- Do NOT use **bold**, *italic*, or any markdown inside JSON values
- Do NOT invent a different word — use exactly "{foreign_word}"
- Return ONLY the JSON object, nothing else

OUTPUT JSON:"""


# ── CJK helpers ───────────────────────────────────────────────────────────────

def _contains_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
                or 0xAC00 <= cp <= 0xD7AF
                or 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF):
            return True
    return False


def _extract_cjk_tokens(text: str) -> list[str]:
    tokens, current = [], []
    for ch in text:
        if _contains_cjk(ch):
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


# ── Agent ──────────────────────────────────────────────────────────────────────

class SentenceAgent(BaseAgent):
    name = "sentence-agent"

    def __init__(
        self,
        rag:    RAGDictionary,
        memory: MemoryStore | None = None,
        vocab:  VocabLoader | None = None,
    ):
        super().__init__()
        self.rag    = rag
        self.memory = memory
        self.vocab  = vocab

    def run(
        self,
        language: str,
        topic:    str,
        user_state: UserState,
        max_retries: int = 3,
    ) -> GeneratedSentence:
        self.clear_logs()
        level    = user_state.level
        is_cjk   = language in CJK_LANGUAGES

        # Build the set of words to avoid
        mastered = user_state.mastered_words[-10:]
        struggled_words, recent_words = [], []
        if self.memory:
            struggled_words = self.memory.get_struggled_words()
            recent_words    = self.memory.get_recent_words(8)
        avoid_set = set(mastered + struggled_words + recent_words)

        self.log(f"level={level}, topic={topic}, lang={language}, cjk={is_cjk}")
        self.log(f"Avoiding {len(avoid_set)} words")

        # ── Pick a word from the curated vocab JSON ────────────────────────────
        vocab_entry: VocabEntry | None = None
        if self.vocab:
            vocab_entry = self.vocab.pick_word(language, topic, level, avoid_set)
            if vocab_entry:
                self.log(
                    f"[vocab] picked '{vocab_entry.word}' "
                    f"({vocab_entry.romanization or 'no romanization'}) "
                    f"= '{vocab_entry.english}' from JSON"
                )
            else:
                self.log("[vocab] no matching entry — falling back to LLM word selection")
        else:
            self.log("[vocab] VocabLoader not available — LLM selects word")

        last_error = None
        for attempt in range(1, max_retries + 1):
            self.log(f"LLM call attempt {attempt}/{max_retries}...")
            try:
                if vocab_entry:
                    # ── Vocab-first path (preferred) ───────────────────────────
                    # Word, romanization, and meaning are all known — LLM writes sentence only
                    rom_hint = vocab_entry.romanization or "(no romanization needed)"
                    prompt = SENTENCE_PROMPT.format(
                        language=language,
                        topic=topic,
                        foreign_word=vocab_entry.word,
                        romanization_hint=rom_hint,
                        english_meaning=vocab_entry.english,
                        pos=vocab_entry.pos,
                        difficulty=vocab_entry.difficulty,
                    )
                    raw  = self._call(prompt, max_tokens=600)
                    data = self._parse_json(raw)

                    # Strip markdown
                    for key in ("foreign_word", "romanization", "english_meaning", "sentence"):
                        if key in data and isinstance(data[key], str):
                            data[key] = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", data[key]).strip()

                    # Override fields with authoritative values from JSON
                    # (model may have paraphrased or changed them)
                    data["foreign_word"]    = vocab_entry.word
                    data["romanization"]    = vocab_entry.romanization
                    data["english_meaning"] = vocab_entry.english
                    data["part_of_speech"]  = vocab_entry.pos
                    data["difficulty"]      = vocab_entry.difficulty
                    data.setdefault("example_context", "")

                else:
                    # ── LLM fallback path (no vocab entry available) ───────────
                    # Build a RAG-enriched prompt asking the LLM to pick a word
                    rag_results = self.rag.lookup(topic, language, top_k=5)
                    rag_context = (
                        ", ".join(f"{e.foreign_word}={e.english_meaning}" for e in rag_results)
                        or "none yet"
                    )
                    rom_label = ROMANIZATION_LABEL.get(language, "")
                    script    = CJK_SCRIPT_NAME.get(language, language)
                    cjk_instruction = (
                        f"IMPORTANT: foreign_word MUST be in {script}, NOT romanization. "
                        f"Put {rom_label} in the romanization field separately.\n"
                        if is_cjk else
                        'Set "romanization" to "" (empty string).\n'
                    )
                    fallback_prompt = (
                        f"You write language learning sentences. Output ONE JSON object only.\n\n"
                        f"Write an English sentence about '{topic}' for a {level} {language} learner. "
                        f"Embed one real {language} word.\n"
                        f"{cjk_instruction}"
                        f"Known words to avoid: {', '.join(avoid_set) or 'none'}\n"
                        f"RAG context: {rag_context}\n\n"
                        f'Fields: "sentence", "foreign_word", "romanization", '
                        f'"english_meaning", "part_of_speech", "example_context", "difficulty"\n\n'
                        f"Output only the JSON:"
                    )
                    raw  = self._call(fallback_prompt, max_tokens=800)
                    data = self._parse_json(raw)

                    for key in ("foreign_word", "romanization", "english_meaning", "sentence"):
                        if key in data and isinstance(data[key], str):
                            data[key] = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", data[key]).strip()
                    data.setdefault("romanization", "")

                    # CJK self-repair
                    if is_cjk and not _contains_cjk(data.get("foreign_word", "")):
                        tokens = _extract_cjk_tokens(data.get("sentence", ""))
                        if tokens:
                            bad = data["foreign_word"]
                            data["foreign_word"] = tokens[0]
                            if _contains_cjk(data.get("romanization", "")):
                                data["romanization"] = ""
                            self.log(f"[self-repair] '{bad}' → '{data['foreign_word']}'")

                    # Romanization fallback
                    if is_cjk and not data.get("romanization", "").strip():
                        fw  = data.get("foreign_word", "")
                        lbl = ROMANIZATION_LABEL.get(language, "romanization")
                        rom = self._call(
                            ROMANIZATION_PROMPT.format(label=lbl, language=language, word=fw),
                            max_tokens=40,
                        ).strip().strip('"').strip("'")
                        data["romanization"] = rom
                        self.log(f"[romanization-fallback] '{rom}' for '{fw}'")

                result = GeneratedSentence(**data)

                # ── Validations ────────────────────────────────────────────────

                # If the model dropped the word from the sentence, try to insert it
                if result.foreign_word not in result.sentence:
                    if vocab_entry:
                        # Construct a minimal valid sentence rather than failing
                        self.log(
                            f"[repair] '{result.foreign_word}' missing from sentence; "
                            f"inserting directly"
                        )
                        result = GeneratedSentence(
                            sentence=f"What does {result.foreign_word} mean?",
                            foreign_word=result.foreign_word,
                            romanization=result.romanization,
                            english_meaning=result.english_meaning,
                            part_of_speech=result.part_of_speech,
                            example_context=result.example_context,
                            difficulty=result.difficulty,
                        )
                    else:
                        raise ValueError(
                            f"foreign_word '{result.foreign_word}' not found in sentence"
                        )

                if is_cjk and not _contains_cjk(result.foreign_word):
                    raise ValueError(
                        f"No CJK characters in foreign_word '{result.foreign_word}'"
                    )

                if is_cjk and not result.romanization.strip():
                    raise ValueError("romanization empty after all fallbacks")

                # ── Register in RAG ────────────────────────────────────────────
                self.rag.add_entry(DictionaryEntry(
                    foreign_word=result.foreign_word,
                    language=language,
                    english_meaning=result.english_meaning,
                    part_of_speech=result.part_of_speech,
                    example_context=result.example_context,
                ))
                self.log(
                    f"✓ '{result.display_word}' → '{result.english_meaning}' "
                    f"(RAG size={self.rag.size()})"
                )
                return result

            except Exception as e:
                last_error = e
                self.log(f"Attempt {attempt} failed: {e} — retrying...")

        raise RuntimeError(
            f"SentenceAgent failed after {max_retries} attempts. Last error: {last_error}"
        )