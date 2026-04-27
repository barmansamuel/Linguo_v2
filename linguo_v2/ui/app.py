"""
ui/app.py — Gradio 6 compatible interface for Linguo

Key Gradio 6 changes applied:
  - gr.update(visible=...) on Row/HTML is unreliable; replaced with always-visible
    components that render empty strings or placeholder HTML instead.
  - Every handler wrapped in try/except so errors surface as readable text,
    not silent red "Error" badges.
  - theme moved to demo.launch() not gr.Blocks().
"""

from __future__ import annotations

import traceback
import gradio as gr

from agents.orchestrator import Orchestrator
from config import SUPPORTED_LANGUAGES, TOPICS

orch = Orchestrator()

# ── Helper formatters ──────────────────────────────────────────────────────────

# BCP-47 language codes for Web Speech API
_LANG_CODES = {
    "Spanish":    "es-ES",
    "French":     "fr-FR",
    "German":     "de-DE",
    "Italian":    "it-IT",
    "Portuguese": "pt-PT",
    "Mandarin":   "zh-CN",
    "Japanese":   "ja-JP",
    "Korean":     "ko-KR",
}


def _highlight_sentence(sentence, foreign_word, display_word, language=""):
    """Replace the native-script token with a clickable TTS button."""
    import re as _re
    lang_code = _LANG_CODES.get(language, "")
    # Escape single quotes so the word is safe inside JS string literals
    safe_word = foreign_word.replace("\\", "\\\\").replace("'", "\\'")
    # Build onclick attribute value
    onclick_val = "speakWord('" + safe_word + "', '" + lang_code + "')"
    # Build hover style values  
    hover_on  = "this.style.background='#2f4ac7'"
    hover_off = "this.style.background='#3b5bdb'"
    button = (
        '<button onclick="' + onclick_val + '"'
        + ' title="Click to hear pronunciation"'
        + ' style="background:#3b5bdb;color:#fff;padding:3px 12px;'
        + 'border-radius:6px;font-weight:600;white-space:nowrap;'
        + 'border:none;cursor:pointer;font-size:inherit;'
        + 'transition:background 0.15s;"'
        + ' onmouseover="' + hover_on + '"'
        + ' onmouseout="' + hover_off + '">'
        + '\U0001f50a ' + display_word + '</button>'
    )
    highlighted = _re.sub(_re.escape(foreign_word), button, sentence, count=1)
    return '<p style="font-size:1.25rem;line-height:1.9;margin:0.5rem 0">' + highlighted + '</p>'


def _error_html(msg: str) -> str:
    return (
        f'<div style="background:#fee2e2;color:#991b1b;padding:12px;'
        f'border-radius:8px;font-family:monospace;white-space:pre-wrap">'
        f'Error: {msg}</div>'
    )


def _vocab_html(state) -> str:
    if not state.vocab:
        return "<p style='color:gray;padding:1rem'>No words yet — start practicing!</p>"
    rows = ""
    for word, rec in state.vocab.items():
        # Dark-theme-safe colors: colored left border + explicit dark text
        if rec.mastered:
            border = "#22c55e"   # green
            badge_bg = "#166534"
            badge_color = "#dcfce7"
            badge_text = "✓ mastered"
        elif rec.attempts > 0:
            border = "#eab308"   # amber
            badge_bg = "#854d0e"
            badge_color = "#fef9c3"
            badge_text = f"{rec.attempts} attempt(s)"
        else:
            border = "#6b7280"   # gray
            badge_bg = "#374151"
            badge_color = "#f3f4f6"
            badge_text = "new"

        rows += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:10px 14px;border-left:4px solid {border};'
            f'border-radius:6px;margin-bottom:6px;'
            f'background:rgba(255,255,255,0.05);">'
            f'<span style="color:#f3f4f6">'
            f'<strong style="color:#ffffff">{word}</strong>'
            f' = {rec.meaning}'
            f' <em style="color:#9ca3af">({rec.lang})</em>'
            f'</span>'
            f'<span style="font-size:0.78rem;font-weight:500;padding:2px 8px;'
            f'border-radius:4px;background:{badge_bg};color:{badge_color}">'
            f'{badge_text}</span>'
            f'</div>'
        )
    return rows


# ── Action handlers ────────────────────────────────────────────────────────────

def handle_generate(language: str, topic: str):
    try:
        sentence_obj, logs = orch.generate_sentence(language, topic)
        html = _highlight_sentence(sentence_obj.sentence, sentence_obj.foreign_word, sentence_obj.display_word, language)
        state = orch.user_state
        stats = (
            f"Words seen: {state.total_seen}  |  "
            f"Mastered: {state.mastered_count}  |  "
            f"Streak: {state.streak}  |  "
            f"Level: {state.level}"
        )
        return html, "", "", stats, "\n".join(logs)
    except Exception as e:
        tb = traceback.format_exc()
        err = _error_html(f"{e}\n\n{tb}")
        return err, "", "", "Error — see sentence area", tb


def handle_check(guess: str):
    if not guess.strip():
        return "", "", ""
    try:
        result, logs = orch.check_answer(guess.strip())
        color = "#dcfce7" if result.correct else "#fee2e2"
        text_color = "#166534" if result.correct else "#991b1b"
        icon = "✓" if result.correct else "✗"
        fb_html = (
            f'<div style="background:{color};color:{text_color};padding:12px 16px;'
            f'border-radius:8px;font-size:1rem">'
            f'<strong>{icon}</strong> {result.feedback}'
            f'</div>'
        )
        state = orch.user_state
        stats = (
            f"Words seen: {state.total_seen}  |  "
            f"Mastered: {state.mastered_count}  |  "
            f"Streak: {state.streak}  |  "
            f"Level: {state.level}"
        )
        return fb_html, stats, "\n".join(logs)
    except Exception as e:
        tb = traceback.format_exc()
        return _error_html(f"{e}\n\n{tb}"), "", tb


def handle_hint():
    try:
        hint, logs = orch.get_hint()
        hint_html = (
            f'<div style="background:#ede9fe;color:#4c1d95;padding:10px 14px;'
            f'border-radius:8px;font-size:0.95rem">'
            f'<strong>Hint:</strong> {hint}</div>'
        )
        return hint_html, "\n".join(logs)
    except Exception as e:
        tb = traceback.format_exc()
        return _error_html(f"{e}\n\n{tb}"), tb


def handle_vocab():
    try:
        return _vocab_html(orch.user_state)
    except Exception as e:
        return _error_html(str(e))


def handle_progress():
    try:
        analysis, logs = orch.get_progress()
        state = orch.user_state
        review = ", ".join(analysis.get("words_to_review", [])) or "none"
        adjust = analysis.get("difficulty_adjustment", "maintain")
        adj_color = {"increase": "#dcfce7", "decrease": "#fee2e2", "maintain": "#f1f5f9"}.get(adjust, "#f1f5f9")
        html = (
            f'<div style="padding:0.5rem 0">'
            f'<p style="font-size:1.05rem;margin-bottom:1rem">{analysis.get("summary", "")}</p>'
            f'<table style="width:100%;border-collapse:collapse;font-size:0.95rem">'
            f'<tr><td style="padding:8px;color:#666;width:220px">Words to review</td>'
            f'<td style="padding:8px"><strong>{review}</strong></td></tr>'
            f'<tr><td style="padding:8px;color:#666">Difficulty</td>'
            f'<td style="padding:8px"><span style="background:{adj_color};padding:2px 10px;border-radius:6px">{adjust}</span></td></tr>'
            f'<tr><td style="padding:8px;color:#666">Current level</td>'
            f'<td style="padding:8px"><strong>{state.level}</strong></td></tr>'
            f'<tr><td style="padding:8px;color:#666">Progress</td>'
            f'<td style="padding:8px"><strong>{state.mastered_count} / {state.total_seen}</strong> words mastered</td></tr>'
            f'</table></div>'
        )
        return html, "\n".join(logs)
    except Exception as e:
        tb = traceback.format_exc()
        return _error_html(f"{e}\n\n{tb}"), tb




def _persistence_banner() -> str:
    """Show what was restored from SQLite on startup."""
    state = orch.user_state
    db_size = orch.db.read_query('SELECT COUNT(*) as n FROM vocab')[0]['n']
    rag_size = orch.rag.size()
    if db_size == 0:
        return ""
    return (
        f'<div style="background:#f0fdf4;color:#166534;padding:10px 16px;'
        f'border-radius:8px;font-size:0.9rem;margin-bottom:0.5rem">'
        f'✓ Restored from previous session: '
        f'<strong>{db_size}</strong> words in vocab, '
        f'<strong>{state.mastered_count}</strong> mastered, '
        f'<strong>{rag_size}</strong> RAG entries loaded'
        f'</div>'
    )


def handle_reset():
    try:
        orch.reset_session()
        state = orch.user_state
        stats = (
            f"Words seen: {state.total_seen}  |  "
            f"Mastered: {state.mastered_count}  |  "
            f"Streak: {state.streak}  |  "
            f"Level: {state.level}"
        )
        return (
            "<p style='color:gray'>Session reset. Press 'New sentence' to begin.</p>",
            "", "", stats,
            "[orchestrator] session reset. Vocab and mastery loaded from SQLite."
        )
    except Exception as e:
        tb = traceback.format_exc()
        return _error_html(f"{e}\n\n{tb}"), "", "", "Error", tb

# ── Layout ─────────────────────────────────────────────────────────────────────

def launch_app():
    speech_js = """
<script>
function speakWord(word, langCode) {
    if (!window.speechSynthesis) {
        alert("Your browser does not support text-to-speech.");
        return;
    }
    // Cancel any currently playing speech
    window.speechSynthesis.cancel();

    var utter = new SpeechSynthesisUtterance(word);
    utter.lang = langCode;
    utter.rate = 0.85;   // slightly slower for clarity
    utter.pitch = 1.0;

    // Try to find a voice matching the language
    var voices = window.speechSynthesis.getVoices();
    var match = voices.find(function(v) {
        return v.lang === langCode || v.lang.startsWith(langCode.split("-")[0]);
    });
    if (match) { utter.voice = match; }

    window.speechSynthesis.speak(utter);
}

// Voices load asynchronously on some browsers — pre-load them
if (window.speechSynthesis) {
    window.speechSynthesis.getVoices();
    window.speechSynthesis.onvoiceschanged = function() {
        window.speechSynthesis.getVoices();
    };
}
</script>
"""

    with gr.Blocks(title="Linguo") as demo:
        gr.HTML(speech_js)   # inject TTS script once on page load
        gr.Markdown("# Linguo\n*Learn vocabulary through context — one word at a time*")

        with gr.Tabs():

            # ── Practice tab ───────────────────────────────────────────────────
            with gr.Tab("Practice"):
                with gr.Row():
                    lang_dd  = gr.Dropdown(SUPPORTED_LANGUAGES, value="Spanish", label="Language")
                    topic_dd = gr.Dropdown(TOPICS, value="everyday life", label="Topic")
                    gen_btn  = gr.Button("New sentence", variant="primary")

                stats_box     = gr.Textbox(label="Session stats", interactive=False, lines=1)
                sentence_html = gr.HTML("<p style='color:gray'>Press 'New sentence' to begin.</p>")
                feedback_html = gr.HTML()
                hint_html     = gr.HTML()

                with gr.Row():
                    guess_input = gr.Textbox(
                        placeholder="Type the English meaning...",
                        label="Your guess",
                        scale=4,
                        interactive=True,
                    )
                    with gr.Column(scale=1, min_width=80):
                        check_btn = gr.Button("Check", variant="primary")
                        hint_btn  = gr.Button("Hint")

                with gr.Accordion("Agent logs", open=False):
                    log_box = gr.Textbox(lines=8, interactive=False, label="")

                gen_btn.click(
                    handle_generate,
                    inputs=[lang_dd, topic_dd],
                    outputs=[sentence_html, feedback_html, hint_html, stats_box, log_box],
                )
                check_btn.click(
                    handle_check,
                    inputs=[guess_input],
                    outputs=[feedback_html, stats_box, log_box],
                )
                hint_btn.click(
                    handle_hint,
                    outputs=[hint_html, log_box],
                )
                guess_input.submit(
                    handle_check,
                    inputs=[guess_input],
                    outputs=[feedback_html, stats_box, log_box],
                )

            # ── Vocabulary tab ─────────────────────────────────────────────────
            with gr.Tab("Vocabulary"):
                refresh_btn = gr.Button("Refresh")
                vocab_html  = gr.HTML("<p style='color:gray;padding:1rem'>No words yet — start practicing!</p>")
                refresh_btn.click(handle_vocab, outputs=[vocab_html])

            # ── Progress tab ───────────────────────────────────────────────────
            with gr.Tab("Progress"):
                analyze_btn   = gr.Button("Analyze my progress", variant="primary")
                progress_html = gr.HTML()
                with gr.Accordion("Agent logs", open=False):
                    prog_log_box = gr.Textbox(lines=8, interactive=False, label="")
                analyze_btn.click(
                    handle_progress,
                    outputs=[progress_html, prog_log_box],
                )

    demo.launch(theme=gr.themes.Soft())
