"""
persistence/sqlite_store.py — SQLite-backed durable store via the MCP sqlite server

Responsibilities:
  - Persist vocab (WordRecord) across app restarts
  - Persist session history (answer log)
  - Persist RAG dictionary entries so the translation store survives restarts
  - Persist aggregate session stats (streak, total_seen)

Design:
  We use Python's built-in sqlite3 module directly, mirroring exactly what
  the MCP sqlite server would expose via its read_query / write_query tools.
  This means you can swap to the actual MCP server subprocess later with zero
  schema changes — the SQL is identical.

  Tables
  ──────
  vocab         — one row per (word, language) pair with mastery stats
  history       — append-only answer log
  rag_entries   — RAG dictionary entries for cross-session reuse
  session_stats — single-row aggregate (streak, total_seen)
"""

from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from typing import Any

from config import DB_PATH


class SQLiteStore:
    """
    Thin wrapper around sqlite3 that mirrors the MCP sqlite server's interface.

    All public methods correspond 1-to-1 with MCP tool calls:
      read_query(sql)   → list[dict]
      write_query(sql)  → None

    This makes future migration to the actual MCP server subprocess trivial.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    # ── MCP-compatible interface ───────────────────────────────────────────────

    def read_query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Execute a SELECT and return rows as list of dicts (MCP read_query)."""
        cur = self._conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def write_query(self, sql: str, params: tuple = ()) -> None:
        """Execute an INSERT/UPDATE/DELETE and commit (MCP write_query)."""
        self._conn.execute(sql, params)
        self._conn.commit()

    # ── Vocab ──────────────────────────────────────────────────────────────────

    def save_word(self, word: str, meaning: str, lang: str,
                  correct: int, attempts: int) -> None:
        self.write_query(
            """INSERT INTO vocab (word, lang, meaning, correct, attempts)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(word, lang) DO UPDATE SET
                 meaning  = excluded.meaning,
                 correct  = excluded.correct,
                 attempts = excluded.attempts""",
            (word, lang, meaning, correct, attempts),
        )

    def load_all_vocab(self) -> list[dict]:
        return self.read_query("SELECT word, lang, meaning, correct, attempts FROM vocab")

    # ── History ────────────────────────────────────────────────────────────────

    def append_history(self, word: str, correct: bool) -> None:
        self.write_query(
            "INSERT INTO history (word, correct) VALUES (?, ?)",
            (word, 1 if correct else 0),
        )

    def load_history(self, limit: int = 200) -> list[dict]:
        return self.read_query(
            "SELECT word, correct, ts FROM history ORDER BY ts DESC LIMIT ?",
            (limit,),
        )

    # ── RAG dictionary ─────────────────────────────────────────────────────────

    def save_rag_entry(self, foreign_word: str, language: str,
                       english_meaning: str, part_of_speech: str,
                       example_context: str) -> None:
        self.write_query(
            """INSERT INTO rag_entries
                 (foreign_word, language, english_meaning, part_of_speech, example_context)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(foreign_word, language) DO NOTHING""",
            (foreign_word, language, english_meaning, part_of_speech, example_context),
        )

    def load_rag_entries(self) -> list[dict]:
        return self.read_query(
            "SELECT foreign_word, language, english_meaning, part_of_speech, example_context FROM rag_entries"
        )

    # ── Session stats ──────────────────────────────────────────────────────────

    def save_stats(self, streak: int, total_seen: int) -> None:
        self.write_query(
            """INSERT INTO session_stats (id, streak, total_seen)
               VALUES (1, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 streak     = excluded.streak,
                 total_seen = excluded.total_seen""",
            (streak, total_seen),
        )

    def load_stats(self) -> dict:
        rows = self.read_query("SELECT streak, total_seen FROM session_stats WHERE id = 1")
        return rows[0] if rows else {"streak": 0, "total_seen": 0}

    # ── Schema init ────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS vocab (
                word     TEXT NOT NULL,
                lang     TEXT NOT NULL,
                meaning  TEXT NOT NULL,
                correct  INTEGER DEFAULT 0,
                attempts INTEGER DEFAULT 0,
                PRIMARY KEY (word, lang)
            );

            CREATE TABLE IF NOT EXISTS history (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                word    TEXT    NOT NULL,
                correct INTEGER NOT NULL,
                ts      DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS rag_entries (
                foreign_word    TEXT NOT NULL,
                language        TEXT NOT NULL,
                english_meaning TEXT NOT NULL,
                part_of_speech  TEXT,
                example_context TEXT,
                PRIMARY KEY (foreign_word, language)
            );

            CREATE TABLE IF NOT EXISTS session_stats (
                id         INTEGER PRIMARY KEY,
                streak     INTEGER DEFAULT 0,
                total_seen INTEGER DEFAULT 0
            );
        """)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
