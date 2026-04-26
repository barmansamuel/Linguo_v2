"""
api/routes.py — Optional FastAPI REST interface

Provides the same functionality as the Gradio UI via HTTP,
useful if you want to build your own frontend or integrate with other services.

Start with:
    uvicorn api.routes:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.orchestrator import Orchestrator

app = FastAPI(title="Linguo API", version="1.0.0")

# One orchestrator per process (in production, use a session store / Redis)
_orchestrator = Orchestrator()


# ── Request / response schemas ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    language: str = "Spanish"
    topic:    str = "everyday life"

class AnswerRequest(BaseModel):
    guess: str

class ProgressResponse(BaseModel):
    summary:               str
    words_to_review:       list[str]
    difficulty_adjustment: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/generate")
def generate(req: GenerateRequest):
    """Generate a new sentence with an embedded foreign word."""
    try:
        sentence, logs = _orchestrator.generate_sentence(req.language, req.topic)
        return {
            "sentence":        sentence.sentence,
            "foreign_word":    sentence.foreign_word,
            "part_of_speech":  sentence.part_of_speech,
            "difficulty":      sentence.difficulty,
            "logs":            logs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/answer")
def answer(req: AnswerRequest):
    """Submit a guess and receive evaluation feedback."""
    try:
        result, logs = _orchestrator.check_answer(req.guess)
        state = _orchestrator.user_state
        return {
            "correct":       result.correct,
            "feedback":      result.feedback,
            "score":         result.score,
            "streak":        state.streak,
            "mastered_count": state.mastered_count,
            "logs":          logs,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/hint")
def hint():
    """Get a contextual hint for the current sentence."""
    try:
        text, logs = _orchestrator.get_hint()
        return {"hint": text, "logs": logs}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/progress")
def progress():
    """Get an AI-generated progress summary and recommendations."""
    analysis, logs = _orchestrator.get_progress()
    return {**analysis, "logs": logs}


@app.get("/vocab")
def vocab():
    """Return the user's full vocabulary record."""
    state = _orchestrator.user_state
    return {
        "total_seen":    state.total_seen,
        "mastered_count": state.mastered_count,
        "level":         state.level,
        "streak":        state.streak,
        "vocab": {
            word: {
                "meaning":  rec.meaning,
                "lang":     rec.lang,
                "correct":  rec.correct,
                "attempts": rec.attempts,
                "mastered": rec.mastered,
                "accuracy": rec.accuracy,
            }
            for word, rec in state.vocab.items()
        },
    }


@app.post("/reset")
def reset():
    """Reset the session (keeps RAG dictionary)."""
    _orchestrator.reset_session()
    return {"status": "reset"}
