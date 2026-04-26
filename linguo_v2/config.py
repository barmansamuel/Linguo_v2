"""
config.py — Shared configuration and constants
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Ollama ─────────────────────────────────────────────────────────────────────
# Ollama runs locally — no API key needed.
# Install: https://ollama.com  →  ollama pull llama3.2
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("LINGUO_MODEL", "llama3.2")   # any model you have pulled

# ── Supported languages ────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = [
    "Spanish", "French", "German", "Italian",
    "Portuguese", "Japanese", "Mandarin", "Korean",
]

# ── Topics ─────────────────────────────────────────────────────────────────────
TOPICS = [
    "everyday life", "food and dining", "travel",
    "work and office", "nature and weather",
    "emotions and feelings", "shopping",
]

# ── Difficulty / level thresholds ──────────────────────────────────────────────
LEVEL_THRESHOLDS = {
    "beginner":     (0, 4),    # 0–4 mastered words
    "intermediate": (5, 14),   # 5–14 mastered words
    "advanced":     (15, 9999),
}

# ── Mastery definition ─────────────────────────────────────────────────────────
MASTERY_CORRECT_THRESHOLD = 2   # correct answers needed to mark a word mastered

# ── RAG ────────────────────────────────────────────────────────────────────────
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 5))
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", 0.75))
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Persistence ────────────────────────────────────────────────────────────────
# SQLite database file path (created automatically on first run)
DB_PATH = os.getenv("LINGUO_DB_PATH", "linguo.db")

# Max recent entries kept in the in-session memory knowledge graph
MEMORY_MAX_RECENT = int(os.getenv("MEMORY_MAX_RECENT", 20))
