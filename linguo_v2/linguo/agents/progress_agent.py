"""
agents/progress_agent.py — Tracks progress and generates adaptive session summaries

Now receives a memory_summary from MemoryStore so it has richer session context:
  - Which words the user struggled with this session
  - Which topics were covered
  - Recent activity pattern
"""

from __future__ import annotations

from state.models import UserState
from agents.base import BaseAgent


PROGRESS_PROMPT = """You are a progress tracker inside a language learning multi-agent system.

Here is the user's current vocabulary data:
{vocab_summary}

Session stats:
- Current streak:   {streak}
- Total words seen: {total_seen}
- Mastered words:   {mastered_count}
- Current level:    {level}

In-session memory context:
- Words struggled with this session: {struggled_words}
- Topics covered this session:       {topics_covered}
- Most recently seen words:          {recent_words}

Tasks:
1. Write a 1-sentence encouraging progress summary that references the session context.
2. Identify up to 3 words the user should review (prioritize struggled words, then lowest accuracy).
3. Suggest a difficulty adjustment: "increase", "maintain", or "decrease".

Return ONLY valid JSON, no markdown, no explanation:
{{
  "summary":               "<1-sentence progress summary>",
  "words_to_review":       ["word1", "word2"],
  "difficulty_adjustment": "increase | maintain | decrease"
}}"""


class ProgressAgent(BaseAgent):
    name = "progress-agent"

    def run(self, user_state: UserState, memory_summary: dict | None = None) -> dict:
        self.clear_logs()
        self.log(f"Analyzing {len(user_state.vocab)} words, level={user_state.level}")

        mem = memory_summary or {}
        struggled = mem.get("struggled_words", [])
        topics    = mem.get("topics_covered", [])
        recent    = mem.get("recent_words", [])

        if struggled:
            self.log(f"[memory] struggled words this session: {struggled}")

        rows = []
        for word, rec in user_state.vocab.items():
            rows.append(
                f"  {word} ({rec.lang}): {rec.attempts} attempts, "
                f"{rec.correct} correct, accuracy={rec.accuracy}, "
                f"mastered={rec.mastered}"
            )
        vocab_summary = "\n".join(rows) if rows else "No words yet."

        prompt = PROGRESS_PROMPT.format(
            vocab_summary=vocab_summary,
            streak=user_state.streak,
            total_seen=user_state.total_seen,
            mastered_count=user_state.mastered_count,
            level=user_state.level,
            struggled_words=", ".join(struggled) or "none",
            topics_covered=", ".join(topics) or "none",
            recent_words=", ".join(recent) or "none",
        )

        raw = self._call(prompt, max_tokens=400)
        data = self._parse_json(raw)
        self.log(f"Adjustment recommendation: {data.get('difficulty_adjustment')}")
        return data
