"""
rag/dictionary.py — RAG-backed translation dictionary

Architecture:
  - A FAISS index stores embeddings of (foreign_word, language, english_meaning) entries.
  - On lookup, the query word + language is embedded and the top-K most similar
    entries are returned, filtered by a similarity threshold.
  - New words are added to the index automatically as the session progresses,
    so the dictionary grows with the user's vocabulary.

This simulates a translation dictionary that can be searched semantically
(useful for morphological variants, synonyms, and closely related terms).
"""

from __future__ import annotations

import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config import RAG_TOP_K, RAG_SIMILARITY_THRESHOLD, EMBEDDING_MODEL


@dataclass
class DictionaryEntry:
    foreign_word:    str
    language:        str
    english_meaning: str
    part_of_speech:  str
    example_context: str


@dataclass
class RAGDictionary:
    """
    Semantic translation dictionary backed by a FAISS vector index.

    Usage:
        rag = RAGDictionary()
        rag.add_entry(entry)
        results = rag.lookup("gato", "Spanish")
    """
    _entries: list[DictionaryEntry]   = field(default_factory=list)
    _index:   object                  = field(default=None, repr=False)
    _embedder: object                 = field(default=None, repr=False)

    def __post_init__(self):
        self._load_embedder()
        self._rebuild_index()

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_entry(self, entry: DictionaryEntry) -> None:
        """Add a new word to the dictionary and update the FAISS index."""
        # Deduplicate by (word, language)
        key = (entry.foreign_word.lower(), entry.language)
        existing_keys = {(e.foreign_word.lower(), e.language) for e in self._entries}
        if key in existing_keys:
            return
        self._entries.append(entry)
        self._rebuild_index()

    def lookup(self, word: str, language: str, top_k: int = RAG_TOP_K) -> list[DictionaryEntry]:
        """
        Semantic lookup: returns up to top_k entries whose embedding
        is closest to the query word+language string.
        """
        if self._index is None or len(self._entries) == 0:
            return []

        query_vec = self._embed(f"{word} ({language})")
        distances, indices = self._index.search(
            np.array([query_vec], dtype="float32"), min(top_k, len(self._entries))
        )
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            # FAISS returns L2 distances; convert to cosine-like similarity
            similarity = 1 / (1 + dist)
            if similarity >= RAG_SIMILARITY_THRESHOLD:
                results.append(self._entries[idx])
        return results

    def exact_lookup(self, word: str, language: str) -> Optional[DictionaryEntry]:
        """Direct exact-match lookup (O(n), used for known words)."""
        for entry in self._entries:
            if entry.foreign_word.lower() == word.lower() and entry.language == language:
                return entry
        return None

    def size(self) -> int:
        return len(self._entries)

    def export_json(self, path: str) -> None:
        """Persist the dictionary to disk as JSON."""
        data = [
            {
                "foreign_word":    e.foreign_word,
                "language":        e.language,
                "english_meaning": e.english_meaning,
                "part_of_speech":  e.part_of_speech,
                "example_context": e.example_context,
            }
            for e in self._entries
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def import_json(self, path: str) -> None:
        """Load a previously exported dictionary from disk."""
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            self.add_entry(DictionaryEntry(**item))

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_embedder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        except ImportError:
            # Graceful fallback: use a trivial hash-based "embedding"
            print("[RAG] sentence-transformers not installed — using fallback embedder.")
            self._embedder = None

    def _embed(self, text: str) -> np.ndarray:
        if self._embedder is not None:
            vec = self._embedder.encode(text, normalize_embeddings=True)
            return vec.astype("float32")
        # Fallback: random-ish deterministic vector from hash
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.random(384).astype("float32")
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _rebuild_index(self) -> None:
        if not self._entries:
            self._index = None
            return
        try:
            import faiss  # type: ignore
        except ImportError:
            self._index = None
            return

        vecs = np.array([
            self._embed(f"{e.foreign_word} ({e.language})")
            for e in self._entries
        ], dtype="float32")

        dim = vecs.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vecs)
        self._index = index
