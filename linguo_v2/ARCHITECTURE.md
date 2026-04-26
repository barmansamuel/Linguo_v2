# Linguo — Architecture & Developer Guide

## Overview

Linguo is an agentic language learning app that teaches vocabulary through contextual guessing.
A sentence is presented with one foreign word embedded inline; the user guesses its English meaning.
A multi-agent workflow handles sentence generation, answer evaluation, hinting, and progress tracking.
A RAG (Retrieval-Augmented Generation) vector dictionary grows with each session and enriches every agent call.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│            Gradio (ui/app.py)  OR  FastAPI (api/routes.py)      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Orchestrator                             │
│                    agents/orchestrator.py                       │
│                                                                 │
│   generate_sentence() → check_answer() → get_hint()            │
│                       → get_progress()                          │
└────┬──────────────┬──────────────┬──────────────┬──────────────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
┌─────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐
│Sentence │  │Evaluator  │  │  Hint    │  │  Progress    │
│ Agent   │  │  Agent    │  │  Agent   │  │   Agent      │
└────┬────┘  └─────┬─────┘  └────┬─────┘  └──────┬───────┘
     │             │             │               │
     └──────┬──────┘             │               │
            │                   │               │
            ▼                   ▼               │
┌─────────────────────────────────────┐         │
│         RAG Dictionary              │         │
│         rag/dictionary.py           │         │
│                                     │         │
│  FAISS vector index                 │         │
│  sentence-transformers embeddings   │         │
│  Semantic lookup + exact lookup     │         │
└─────────────────────────────────────┘         │
                                                │
            ┌───────────────────────────────────┘
            ▼
┌─────────────────────────────────────┐
│           UserState                 │
│         state/models.py             │
│                                     │
│  vocab: dict[word → WordRecord]     │
│  streak, total_seen, history        │
│  level (computed from mastery)      │
└─────────────────────────────────────┘
```

---

## Agent Workflow

### Turn 1 — Generate sentence

```
User selects language + topic
         │
         ▼
Orchestrator.generate_sentence(language, topic)
         │
         ├─► RAGDictionary.lookup(topic, language)
         │       Returns semantically related known words as context
         │
         ├─► SentenceAgent.run(language, topic, user_state)
         │       Builds prompt with: level, topic, avoid_words, rag_context
         │       Calls Anthropic API → GeneratedSentence (JSON)
         │
         ├─► RAGDictionary.add_entry(new word)
         │       Embeds and indexes the new word
         │
         └─► UserState.record_word(word, meaning, lang)
                 Adds to vocab if not already tracked
```

### Turn 2 — Evaluate answer

```
User types guess and submits
         │
         ▼
Orchestrator.check_answer(guess)
         │
         ├─► RAGDictionary.lookup(foreign_word, language)
         │       Returns synonym candidates for flexible evaluation
         │
         ├─► EvaluatorAgent.run(language, word, correct_meaning, guess)
         │       Prompt includes synonyms from RAG
         │       Calls Anthropic API → EvaluationResult (correct, feedback, score)
         │
         └─► UserState.record_answer(word, correct)
                 Updates correct count, attempts, streak
```

### Optional — Get hint

```
User clicks Hint
         │
         ▼
Orchestrator.get_hint()
         │
         ├─► RAGDictionary.exact_lookup(word, language)
         │       Retrieves part_of_speech + example_context for richer hints
         │
         └─► HintAgent.run(language, word, correct_meaning, sentence)
                 Returns a subtle hint string (≤20 words)
```

### Optional — Progress analysis

```
User opens Progress tab
         │
         ▼
Orchestrator.get_progress()
         │
         └─► ProgressAgent.run(user_state)
                 Summarizes vocab, identifies weak words,
                 recommends difficulty adjustment
```

---

## RAG Dictionary Design

The RAG dictionary is the shared knowledge base across all agents.

### Storage
- Entries are stored as `DictionaryEntry` dataclass instances in memory.
- A FAISS `IndexFlatL2` holds the corresponding L2-normalized embeddings.
- The index is rebuilt on every `add_entry()` call (acceptable at small scale; switch to `IndexIVFFlat` for 10k+ words).

### Embeddings
- Model: `all-MiniLM-L6-v2` via `sentence-transformers` (384-dimensional).
- Query format: `"<word> (<language>)"` — e.g., `"gato (Spanish)"`.
- Fallback: deterministic hash-based random vector if `sentence-transformers` is not installed.

### Similarity
- FAISS returns L2 distances; similarity is approximated as `1 / (1 + L2_distance)`.
- Entries below `RAG_SIMILARITY_THRESHOLD` (default 0.75) are filtered out.

### Usage per agent
| Agent    | Lookup type    | Purpose                              |
|----------|----------------|--------------------------------------|
| Sentence | semantic       | Avoid re-teaching already-known words |
| Evaluator| semantic       | Find acceptable synonym answers      |
| Hint     | exact          | Retrieve POS + example context       |

---

## Adaptive Difficulty

Level is computed automatically from the user's mastered word count:

| Level        | Mastered words |
|--------------|----------------|
| beginner     | 0 – 4          |
| intermediate | 5 – 14         |
| advanced     | 15 +           |

The `SentenceAgent` prompt includes the current level and adjusts vocabulary accordingly.
The `ProgressAgent` additionally recommends `"increase"`, `"maintain"`, or `"decrease"`.

---

## File Structure

```
linguo/
├── main.py                   # Entry point — launches Gradio
├── config.py                 # API keys, model, constants, thresholds
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── __init__.py
│   ├── base.py               # BaseAgent: API calls, JSON parsing, logging
│   ├── orchestrator.py       # Coordinates all agents + state
│   ├── sentence_agent.py     # Generates contextual sentences
│   ├── evaluator_agent.py    # Grades user answers
│   ├── hint_agent.py         # Produces subtle hints
│   └── progress_agent.py     # Tracks progress + recommends adjustments
│
├── rag/
│   ├── __init__.py
│   └── dictionary.py         # FAISS-backed semantic translation dictionary
│
├── state/
│   ├── __init__.py
│   └── models.py             # Pydantic models: UserState, WordRecord, etc.
│
├── api/
│   ├── __init__.py
│   └── routes.py             # Optional FastAPI REST interface
│
└── ui/
    ├── __init__.py
    └── app.py                # Gradio UI
```

---

## Setup & Running

### 1. Install Ollama

Download from https://ollama.com and install for your OS. Then pull a model:

```bash
ollama pull llama3.2          # recommended — fast, strong at multilingual tasks
# alternatives: mistral, gemma3, qwen2.5, phi4
```

Confirm it's running:
```bash
ollama list                   # shows pulled models
curl http://localhost:11434   # should return "Ollama is running"
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if you want a different model or Ollama runs on a non-default port
```

### 4. Run (Gradio UI)

```bash
python main.py
# Opens at http://localhost:7860
```

### 5. Run (FastAPI REST)

```bash
uvicorn api.routes:app --reload
# API docs at http://localhost:8000/docs
```

---

## Extending the App

### Change the model
Set `LINGUO_MODEL` in your `.env` to any model you have pulled via `ollama pull`:
- `llama3.2` — default, well-rounded multilingual performance
- `mistral` — fast, good JSON adherence
- `gemma3` — strong at instruction following
- `qwen2.5` — excellent for Asian language tasks (Japanese, Mandarin, Korean)

### Add a new agent
1. Subclass `BaseAgent` in `agents/`.
2. Implement `run(**kwargs) -> YourReturnType`.
3. Instantiate in `Orchestrator.__init__()` and wire up a new public method.

### Persist the RAG dictionary between sessions
Call `rag.export_json("dictionary.json")` on shutdown and `rag.import_json("dictionary.json")` on startup in `Orchestrator.__init__()`.

### Scale the FAISS index
For large dictionaries (10k+ entries), replace `IndexFlatL2` with:
```python
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, n_clusters)
index.train(all_vectors)
```

### Add streaming responses
`BaseAgent._call_streaming()` is already implemented. Replace `_call()` with it in any agent and yield chunks to the UI via a Gradio generator function.
