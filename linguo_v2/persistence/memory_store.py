"""
persistence/memory_store.py — In-session knowledge graph (MCP memory server pattern)

Mirrors the MCP memory server's entity/relation/observation model:
  - Entities   → named nodes (words, sessions, topics)
  - Relations  → typed edges between entities
  - Observations → timestamped facts attached to entities

This powers agent working memory within a session:
  - Which words the user has struggled with this session
  - Which topics have been covered
  - Recent sentence context (so agents don't repeat themselves)
  - Agent-to-agent context passing (orchestrator writes, agents read)

The interface exactly mirrors the MCP memory server tools:
  create_entities(), create_relations(), add_observations(),
  search_nodes(), open_nodes()

When you install the real MCP memory server, swap this class out with
async calls to the subprocess — the method signatures are identical.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from config import MEMORY_MAX_RECENT


@dataclass
class Entity:
    name:         str
    entity_type:  str                        # "word" | "topic" | "session" | "sentence"
    observations: list[str] = field(default_factory=list)
    created_at:   float     = field(default_factory=time.time)


@dataclass
class Relation:
    from_entity: str
    relation:    str                         # "struggled_with" | "mastered" | "covered_topic" etc.
    to_entity:   str
    created_at:  float = field(default_factory=time.time)


class MemoryStore:
    """
    In-session knowledge graph — mirrors the MCP memory server interface.

    Entity types used by Linguo:
      "word"     — a foreign vocabulary word seen this session
      "topic"    — a conversation topic (food, travel, etc.)
      "sentence" — a generated sentence (recent context)
      "session"  — the current session node (root)

    Relation types:
      "struggled_with"  — user got wrong ≥ 1 time
      "mastered"        — user got right ≥ threshold times
      "covered_topic"   — session covered this topic
      "last_sentence"   — most recent sentence generated
      "avoided_word"    — word agent should not reuse soon
    """

    def __init__(self):
        self._entities:  dict[str, Entity]   = {}
        self._relations: list[Relation]      = []
        self._recent:    list[str]           = []   # entity names in recency order

        # Bootstrap the session root entity
        self.create_entities([{"name": "session", "entityType": "session",
                                "observations": ["Session started"]}])

    # ── MCP memory server interface ────────────────────────────────────────────

    def create_entities(self, entities: list[dict]) -> list[Entity]:
        """
        MCP: create_entities
        Each dict: {name, entityType, observations: [str]}
        """
        created = []
        for e in entities:
            name = e["name"]
            if name not in self._entities:
                entity = Entity(
                    name=name,
                    entity_type=e.get("entityType", "unknown"),
                    observations=list(e.get("observations", [])),
                )
                self._entities[name] = entity
                self._track_recent(name)
            created.append(self._entities[name])
        return created

    def create_relations(self, relations: list[dict]) -> list[Relation]:
        """
        MCP: create_relations
        Each dict: {from, relationType, to}
        """
        created = []
        for r in relations:
            rel = Relation(
                from_entity=r["from"],
                relation=r["relationType"],
                to_entity=r["to"],
            )
            # Deduplicate
            exists = any(
                x.from_entity == rel.from_entity
                and x.relation == rel.relation
                and x.to_entity == rel.to_entity
                for x in self._relations
            )
            if not exists:
                self._relations.append(rel)
            created.append(rel)
        return created

    def add_observations(self, observations: list[dict]) -> None:
        """
        MCP: add_observations
        Each dict: {entityName, contents: [str]}
        """
        for obs in observations:
            name = obs["entityName"]
            if name not in self._entities:
                self.create_entities([{"name": name, "entityType": "unknown"}])
            self._entities[name].observations.extend(obs.get("contents", []))
            self._track_recent(name)

    def search_nodes(self, query: str) -> list[Entity]:
        """
        MCP: search_nodes
        Returns entities whose name, type, or observations contain the query string.
        """
        q = query.lower()
        return [
            e for e in self._entities.values()
            if q in e.name.lower()
            or q in e.entity_type.lower()
            or any(q in obs.lower() for obs in e.observations)
        ]

    def open_nodes(self, names: list[str]) -> list[Entity]:
        """MCP: open_nodes — retrieve specific entities by name."""
        return [self._entities[n] for n in names if n in self._entities]

    # ── Linguo-specific helpers (built on top of the MCP primitives) ───────────

    def record_word_seen(self, word: str, lang: str, meaning: str,
                         sentence: str) -> None:
        """Register that the user just saw a new word."""
        self.create_entities([{
            "name": f"word:{lang}:{word}",
            "entityType": "word",
            "observations": [f"meaning={meaning}", f"first_seen_in='{sentence[:60]}'"],
        }])
        self.create_relations([{
            "from": "session",
            "relationType": "encountered_word",
            "to": f"word:{lang}:{word}",
        }])

    def record_answer(self, word: str, lang: str, correct: bool,
                      guess: str) -> None:
        """Update the word entity with the answer outcome."""
        key = f"word:{lang}:{word}"
        outcome = f"correct_guess='{guess}'" if correct else f"wrong_guess='{guess}'"
        self.add_observations([{"entityName": key, "contents": [outcome]}])

        rel_type = "mastered" if correct else "struggled_with"
        self.create_relations([{"from": "session", "relationType": rel_type, "to": key}])

    def record_topic_covered(self, topic: str) -> None:
        self.create_entities([{"name": f"topic:{topic}", "entityType": "topic",
                                "observations": [f"covered at t={time.time():.0f}"]}])
        self.create_relations([{"from": "session", "relationType": "covered_topic",
                                 "to": f"topic:{topic}"}])

    def record_sentence(self, sentence: str, foreign_word: str) -> None:
        """Keep track of the most recent sentence for context."""
        key = f"sentence:{foreign_word}"
        self.create_entities([{"name": key, "entityType": "sentence",
                                "observations": [sentence]}])
        # Replace previous last_sentence relation
        self._relations = [r for r in self._relations
                           if r.relation != "last_sentence"]
        self.create_relations([{"from": "session", "relationType": "last_sentence",
                                 "to": key}])

    def get_struggled_words(self) -> list[str]:
        """Return foreign words the user got wrong this session."""
        struggled_rels = [r.to_entity for r in self._relations
                          if r.relation == "struggled_with"]
        return [k.split(":", 2)[-1] for k in struggled_rels]

    def get_recent_words(self, n: int = 5) -> list[str]:
        """Return the n most recently encountered foreign words."""
        word_entities = [
            name for name in reversed(self._recent)
            if name.startswith("word:")
        ]
        return [k.split(":", 2)[-1] for k in word_entities[:n]]

    def get_session_summary(self) -> dict:
        """Compact summary used by ProgressAgent for richer context."""
        struggled = self.get_struggled_words()
        recent    = self.get_recent_words(10)
        topics    = [r.to_entity.replace("topic:", "") for r in self._relations
                     if r.relation == "covered_topic"]
        mastered  = [r.to_entity.split(":", 2)[-1] for r in self._relations
                     if r.relation == "mastered"]
        return {
            "struggled_words": struggled,
            "recent_words":    recent,
            "topics_covered":  list(set(topics)),
            "mastered_this_session": mastered,
            "total_entities":  len(self._entities),
        }

    def clear(self) -> None:
        """Reset for a new session (keep entity types, drop data)."""
        self.__init__()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _track_recent(self, name: str) -> None:
        if name in self._recent:
            self._recent.remove(name)
        self._recent.append(name)
        if len(self._recent) > MEMORY_MAX_RECENT:
            self._recent.pop(0)
