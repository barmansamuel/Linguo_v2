# Linguo

**Learn vocabulary through context — one word at a time.**

Linguo is a free, fully local language learning app powered by [Ollama](https://ollama.com). It presents you with an English sentence containing one embedded foreign word, and challenges you to guess its meaning from context. A multi-agent AI workflow generates sentences, evaluates your answers, provides hints, and adapts to your growing vocabulary — all running on your own machine, at no cost.

---

## Features

- **8 languages** — Spanish, French, German, Italian, Portuguese, Japanese, Mandarin, Korean
- **Adaptive difficulty** — automatically adjusts from beginner to advanced as you master words
- **Multi-agent AI** — separate agents for sentence generation, answer evaluation, hinting, and progress analysis
- **RAG dictionary** — a local semantic vector store grows with every word you encounter
- **No API keys, no cost** — runs entirely via Ollama on your own hardware

---

## Requirements

- Python 3.10 or higher
- [Ollama](https://ollama.com) installed and running
- At least one Ollama model pulled (see below)

---

## Installation

### 1. Install Ollama

Download and install Ollama from [https://ollama.com](https://ollama.com), then pull a model:

```bash
ollama pull llama3.2
```

Verify it's working:

```bash
ollama list                  # should show llama3.2
curl http://localhost:11434  # should return: Ollama is running
```

**Recommended models by use case:**

| Model | Best for |
|---|---|
| `llama3.2` | General use — best default |
| `qwen2.5` | Japanese, Mandarin, Korean |
| `mistral` | Fast responses, reliable JSON |
| `gemma3` | Strong instruction following |

### 2. Clone or unzip the project

```bash
unzip linguo_ollama.zip
cd linguo
```

### 3. (Optional) Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

The defaults work out of the box. Edit `.env` only if you want to change the model or Ollama port:

```env
# Ollama server address (default — only change if Ollama runs on a different port or machine)
OLLAMA_BASE_URL=http://localhost:11434

# Model to use — must be pulled first with: ollama pull <model>
LINGUO_MODEL=llama3.2

# Optional: RAG retrieval settings
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.75
```

---

## Running the App

Make sure Ollama is running in the background, then:

```bash
python main.py
```

Open your browser and go to:

```
http://127.0.0.1:7860
```

The terminal will show startup messages — this is normal. The app is ready when you see:

```
* Running on local URL:  http://127.0.0.1:7860
```

---

## How to Use

### Practice tab

1. Select a **language** and **topic** from the dropdowns
2. Click **New sentence** — a sentence appears with one foreign word highlighted in blue
3. Type your guess for what the highlighted word means in English
4. Press **Enter** or click **Check** to submit
5. Click **Hint** if you're stuck — a contextual clue appears without giving away the answer
6. Your session stats (words seen, mastered, streak, level) update after each answer

### Vocabulary tab

Click **Refresh** to see all the words you've encountered this session, color-coded by mastery:

- Green — mastered (answered correctly twice)
- Yellow — in progress
- White — seen but not yet attempted

### Progress tab

Click **Analyze my progress** for an AI-generated summary of your session, including which words to review and whether the difficulty should be adjusted.

---

## Project Structure

```
linguo/
├── main.py                   # Entry point
├── config.py                 # Settings and constants
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── orchestrator.py       # Coordinates all agents
│   ├── sentence_agent.py     # Generates contextual sentences
│   ├── evaluator_agent.py    # Grades your answers
│   ├── hint_agent.py         # Produces subtle hints
│   └── progress_agent.py     # Tracks progress and recommends adjustments
│
├── rag/
│   └── dictionary.py         # FAISS-backed semantic translation dictionary
│
├── state/
│   └── models.py             # Data models (UserState, WordRecord, etc.)
│
├── api/
│   └── routes.py             # Optional FastAPI REST interface
│
├── ui/
│   └── app.py                # Gradio web interface
│
└── tests/
    └── test_linguo.py        # Test suite
```

---

## Optional: REST API

If you want to integrate Linguo into another app or build your own frontend, a FastAPI interface is included:

```bash
pip install fastapi uvicorn
uvicorn api.routes:app --reload
```

API documentation is available at `http://localhost:8000/docs`.

---

## Running Tests

No Ollama instance needed — API calls are mocked:

```bash
pip install pytest
pytest tests/ -v
```

---

## Troubleshooting

**The app opens but clicking "New sentence" shows an error in red**
- Make sure Ollama is running: `curl http://localhost:11434`
- Make sure your chosen model is pulled: `ollama list`
- Check the "Agent logs" accordion in the UI for the full error message

**Ollama is slow to respond**
- Smaller models like `mistral` or `llama3.2` are faster than larger ones
- Close other resource-heavy apps while running

**`sentence-transformers` shows UNEXPECTED warnings on startup**
- These are harmless — the embedding model is loading and can be safely ignored

**Port 7860 is already in use**
- Another Gradio app may be running. Stop it, or edit `main.py` to pass `server_port=7861` to `demo.launch()`
