"""
agents/sentence_agent.py — Generates contextual sentences with an embedded foreign word

CJK handling strategy (Mandarin, Japanese, Korean):
  - Prompt uses a fill-in-the-blank format to strongly guide the model
  - After parsing, a self-repair pass extracts CJK tokens from the sentence
    if the model put romanization in foreign_word instead of native script
  - If romanization is missing a second lightweight call generates it
  - Validation only fails if self-repair also cannot find any CJK in the sentence
"""

from __future__ import annotations

import re
import unicodedata

from rag.dictionary import RAGDictionary, DictionaryEntry
from state.models import GeneratedSentence, UserState
from agents.base import BaseAgent
from persistence.memory_store import MemoryStore


# ── Language metadata ──────────────────────────────────────────────────────────

CJK_LANGUAGES = {"Mandarin", "Japanese", "Korean"}

CJK_ROMANIZATION_LABEL = {
    "Mandarin": "Pinyin (e.g. shuǐ, māo, gǒu)",
    "Japanese": "Romaji (e.g. mizu, neko, hon)",
    "Korean":   "Romanization (e.g. mul, chaek, go-yang-i)",
}

CJK_SCRIPT_NAME = {
    "Mandarin": "Chinese characters (汉字)",
    "Japanese": "Kanji or Kana (漢字・かな)",
    "Korean":   "Hangul (한글)",
}

# Several worked examples per language — randomly chosen on each attempt
CJK_EXAMPLES = {
    "Mandarin": [
        '{"sentence":"I drink 水 every morning.","foreign_word":"水","romanization":"shuǐ","english_meaning":"water","part_of_speech":"noun","example_context":"我每天喝水。","difficulty":"easy"}',
        '{"sentence":"She has a 猫 at home.","foreign_word":"猫","romanization":"māo","english_meaning":"cat","part_of_speech":"noun","example_context":"她有一只猫。","difficulty":"easy"}',
        '{"sentence":"He went to the 市场 to buy vegetables.","foreign_word":"市场","romanization":"shìchǎng","english_meaning":"market","part_of_speech":"noun","example_context":"我去市场买菜。","difficulty":"medium"}',
    ],
    "Japanese": [
        '{"sentence":"He reads a 本 before bed.","foreign_word":"本","romanization":"hon","english_meaning":"book","part_of_speech":"noun","example_context":"本を読む。","difficulty":"easy"}',
        '{"sentence":"The 犬 ran across the park.","foreign_word":"犬","romanization":"inu","english_meaning":"dog","part_of_speech":"noun","example_context":"犬が走る。","difficulty":"easy"}',
        '{"sentence":"She 食べる sushi every Friday.","foreign_word":"食べる","romanization":"taberu","english_meaning":"to eat","part_of_speech":"verb","example_context":"私は食べる。","difficulty":"medium"}',
    ],
    "Korean": [
        '{"sentence":"I saw a 고양이 in the garden.","foreign_word":"고양이","romanization":"go-yang-i","english_meaning":"cat","part_of_speech":"noun","example_context":"고양이가 귀엽다.","difficulty":"easy"}',
        '{"sentence":"She drinks 물 with every meal.","foreign_word":"물","romanization":"mul","english_meaning":"water","part_of_speech":"noun","example_context":"물을 마셔요.","difficulty":"easy"}',
        '{"sentence":"He borrowed a 책 from the library.","foreign_word":"책","romanization":"chaek","english_meaning":"book","part_of_speech":"noun","example_context":"책을 읽어요.","difficulty":"easy"}',
    ],
}

LATIN_EXAMPLE = (
    '{"sentence":"She put the leche in her coffee.","foreign_word":"leche","romanization":"",'
    '"english_meaning":"milk","part_of_speech":"noun","example_context":"La leche está fría.","difficulty":"easy"}'
)


# ── Prompts ────────────────────────────────────────────────────────────────────

CJK_SENTENCE_PROMPT = """You generate language learning sentences. Your output is always a single JSON object and nothing else.

TASK: Write an English sentence about "{topic}" for a {level} learner.
Embed exactly ONE {language} word written in {script_name} into the sentence.

FIELD RULES — follow exactly:
  "sentence"      → The English sentence with the {script_name} word inside it (NOT romanization)
  "foreign_word"  → ONLY the {script_name} characters, e.g. 水 or 猫 or 고양이. NEVER pinyin or romaji here.
  "romanization"  → The pronunciation in {romanization_label}. This is SEPARATE from foreign_word.
  "english_meaning" → The English translation of the foreign word
  "part_of_speech"  → noun / verb / adjective / adverb
  "example_context" → One short sentence in {language} using the word
  "difficulty"      → easy / medium / hard

WRONG (do not do this):
  "sentence": "I drink shui every morning."   ← romanization in sentence
  "foreign_word": "shui"                      ← romanization in foreign_word

CORRECT:
  "sentence": "I drink 水 every morning."     ← native character in sentence
  "foreign_word": "水"                        ← native character in foreign_word
  "romanization": "shuǐ"                     ← pronunciation separately

EXAMPLES:
{examples}

Avoid these words: {avoid_words}

Output only the JSON object for {language}, topic "{topic}", level {level}:"""

LATIN_SENTENCE_PROMPT = """You generate language learning sentences. Your output is always a single JSON object and nothing else.

TASK: Write an English sentence about "{topic}" for a {level} {language} learner.
Replace exactly ONE English word with its genuine {language} translation.

FIELD RULES:
  "sentence"        → English sentence with ONE {language} word embedded
  "foreign_word"    → The {language} word used (must appear in sentence)
  "romanization"    → "" (empty — not needed for {language})
  "english_meaning" → English translation of the foreign word
  "part_of_speech"  → noun / verb / adjective / adverb
  "example_context" → One short {language} sentence using the word
  "difficulty"      → easy / medium / hard

Rules: real {language} words only (no English loanwords like taxi/hotel/pizza).
Avoid these words: {avoid_words}
RAG context (known words): {rag_context}

EXAMPLE:
{example}

Output only the JSON object for {language}, topic "{topic}", level {level}:"""

ROMANIZATION_PROMPT = """Give the {label} (pronunciation) for the {language} word "{word}".
Return only the romanization string, nothing else. No punctuation, no explanation."""


# ── Loanword blocklist ─────────────────────────────────────────────────────────

_LOANWORDS = {
    "sandwich", "taxi", "hotel", "pizza", "café", "cafe", "internet",
    "email", "bus", "radio", "television", "chocolate", "tomato",
    "banana", "mango", "sofa", "pasta", "sushi", "yoga", "karate",
    "sport", "football", "basketball", "tennis", "golf", "bar",
    "menu", "restaurant", "supermarket", "airport", "metro",
}


# ── CJK utilities ──────────────────────────────────────────────────────────────

def _contains_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if (
            0x4E00 <= cp <= 0x9FFF    # CJK Unified Ideographs
            or 0x3400 <= cp <= 0x4DBF # CJK Extension A
            or 0xAC00 <= cp <= 0xD7AF # Hangul syllables
            or 0x3040 <= cp <= 0x309F # Hiragana
            or 0x30A0 <= cp <= 0x30FF # Katakana
            or 0x3000 <= cp <= 0x303F # CJK Symbols/Punctuation
        ):
            return True
    return False


def _extract_cjk_tokens(text: str) -> list[str]:
    """Extract all contiguous runs of CJK characters from text."""
    tokens = []
    current = []
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


def _script_name(language: str) -> str:
    return CJK_SCRIPT_NAME.get(language, language)


# ── Agent ──────────────────────────────────────────────────────────────────────

class SentenceAgent(BaseAgent):
    name = "sentence-agent"

    def __init__(self, rag: RAGDictionary, memory: MemoryStore | None = None):
        super().__init__()
        self.rag    = rag
        self.memory = memory
        self._attempt_counter = 0   # used to rotate examples

    def run(
        self,
        language: str,
        topic: str,
        user_state: UserState,
        max_retries: int = 3,
    ) -> GeneratedSentence:
        self.clear_logs()
        level    = user_state.level
        mastered = user_state.mastered_words[-10:]
        is_cjk   = language in CJK_LANGUAGES

        rag_results = self.rag.lookup(topic, language, top_k=5)
        rag_context = (
            ", ".join(f"{e.foreign_word}={e.english_meaning}" for e in rag_results)
            or "none yet"
        )

        struggled_words, recent_words = [], []
        if self.memory:
            struggled_words = self.memory.get_struggled_words()
            recent_words    = self.memory.get_recent_words(5)
            self.log(f"[memory] {len(struggled_words)} struggled, {len(recent_words)} recent words")

        avoid_set = list(dict.fromkeys(mastered + struggled_words + recent_words))

        self.log(f"level={level}, topic={topic}, lang={language}, cjk={is_cjk}")
        self.log(f"Avoiding {len(avoid_set)} words")

        last_error = None
        for attempt in range(1, max_retries + 1):
            self.log(f"LLM call attempt {attempt}/{max_retries}...")
            try:
                # Rotate examples across attempts so the model sees variety
                if is_cjk:
                    examples_list = CJK_EXAMPLES[language]
                    example_str   = "\n".join(
                        examples_list[(self._attempt_counter + i) % len(examples_list)]
                        for i in range(min(2, len(examples_list)))
                    )
                    prompt = CJK_SENTENCE_PROMPT.format(
                        language=language,
                        topic=topic,
                        level=level,
                        script_name=_script_name(language),
                        romanization_label=CJK_ROMANIZATION_LABEL[language],
                        avoid_words=", ".join(avoid_set) or "none",
                        examples=example_str,
                    )
                else:
                    prompt = LATIN_SENTENCE_PROMPT.format(
                        language=language,
                        topic=topic,
                        level=level,
                        avoid_words=", ".join(avoid_set) or "none",
                        rag_context=rag_context,
                        example=LATIN_EXAMPLE,
                    )

                raw  = self._call(prompt, max_tokens=800)
                data = self._parse_json(raw)

                # Strip markdown bold/italic
                for key in ("foreign_word", "romanization", "english_meaning", "sentence"):
                    if key in data and isinstance(data[key], str):
                        data[key] = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", data[key]).strip()

                data.setdefault("romanization", "")

                # ── CJK self-repair ────────────────────────────────────────────
                # If the model put romanization in foreign_word, try to recover
                # by extracting the CJK token directly from the sentence.
                if is_cjk and not _contains_cjk(data.get("foreign_word", "")):
                    sentence_text = data.get("sentence", "")
                    cjk_tokens    = _extract_cjk_tokens(sentence_text)
                    if cjk_tokens:
                        bad_fw = data["foreign_word"]
                        data["foreign_word"] = cjk_tokens[0]
                        self.log(
                            f"[self-repair] foreign_word was '{bad_fw}' (romanization); "
                            f"extracted '{data['foreign_word']}' from sentence"
                        )
                        # If the model put the native char in romanization by mistake, swap
                        if _contains_cjk(data.get("romanization", "")):
                            data["romanization"] = ""

                # ── Romanization fallback ──────────────────────────────────────
                # If romanization is still empty for CJK, generate it separately
                if is_cjk and not data.get("romanization", "").strip():
                    fw    = data.get("foreign_word", "")
                    label = CJK_ROMANIZATION_LABEL[language]
                    rom   = self._call(
                        ROMANIZATION_PROMPT.format(
                            label=label, language=language, word=fw
                        ),
                        max_tokens=40,
                    ).strip().strip('"').strip("'")
                    data["romanization"] = rom
                    self.log(f"[romanization-fallback] generated '{rom}' for '{fw}'")

                result = GeneratedSentence(**data)

                # ── Validations ────────────────────────────────────────────────

                if result.foreign_word not in result.sentence:
                    raise ValueError(
                        f"foreign_word '{result.foreign_word}' not found in sentence"
                    )

                if is_cjk and not _contains_cjk(result.foreign_word):
                    raise ValueError(
                        f"No CJK characters in foreign_word '{result.foreign_word}' "
                        f"and none could be extracted from sentence"
                    )

                if is_cjk and not result.romanization.strip():
                    raise ValueError("romanization still empty after fallback")

                if result.foreign_word.lower() in _LOANWORDS:
                    raise ValueError(f"'{result.foreign_word}' is a loanword")

                if result.foreign_word.lower() == result.english_meaning.lower():
                    raise ValueError(
                        f"foreign_word identical to english_meaning: '{result.foreign_word}'"
                    )

                # ── Register in RAG ────────────────────────────────────────────
                self.rag.add_entry(DictionaryEntry(
                    foreign_word=result.foreign_word,
                    language=language,
                    english_meaning=result.english_meaning,
                    part_of_speech=result.part_of_speech,
                    example_context=result.example_context,
                ))
                self._attempt_counter += 1
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
