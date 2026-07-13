"""Provenance Guard: signal extractors.

Two independent detectors that estimate how "AI-generated" a piece of text looks:

  * llm_signal:          asks an LLM to judge the text holistically.
  * stylometric_signal:  pure-Python surface statistics (no network, no deps).

Each returns a plain dict so callers can consume either signal on its own.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from typing import Optional

from dotenv import load_dotenv

# Load GROQ_API_KEY (and any other env vars) once at import time.
load_dotenv()

_GROQ_MODEL = "llama-3.3-70b-versatile"

_SYSTEM_PROMPT = (
    "You are a forensic text analyst deciding whether a passage reads as "
    "AI-generated rather than human-written. Look for tell-tale AI signatures: "
    'stock transitions ("furthermore", "it is important to note"), hedged '
    "both-sides framing, uniform sentence rhythm, an absence of lived or "
    "personal specificity, and generic, safe vocabulary. Weigh substance over "
    "surface polish. Formal register ALONE is NOT proof of AI: academic writers "
    "and non-native speakers legitimately write formally, so do not penalize "
    "formality by itself. Respond ONLY with a strict JSON object of the form "
    '{"ai_likelihood": <float 0-1>, "rationale": "<one sentence>"}.'
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a numeric value into the inclusive [low, high] range."""
    return max(low, min(high, value))


def llm_signal(text: str) -> dict:
    """Ask an LLM whether the text reads as AI-generated.

    Measures: a holistic, semantics-aware judgment of AI likelihood that surface
    statistics cannot capture (voice, specificity, cliche density).
    Blind spot: it is a black box that can be confidently wrong, is non-free /
    network-dependent, and may over-flag formal-but-human prose.

    Returns {"ok": True, "score": float, "rationale": str} on success, or
    {"ok": False, "score": None, "rationale": "..."} on any failure. Never raises.
    """
    try:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return {
                "ok": False,
                "score": None,
                "rationale": "llm signal unavailable: missing GROQ_API_KEY",
            }

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=_GROQ_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Assess whether the following text reads as AI-generated.\n\n"
                        f"{text}"
                    ),
                },
            ],
        )

        content = completion.choices[0].message.content
        parsed = json.loads(content)
        score = _clamp(float(parsed["ai_likelihood"]))
        rationale = str(parsed.get("rationale", "")).strip()
        return {"ok": True, "score": score, "rationale": rationale}
    except Exception as exc:  # never raise; degrade gracefully
        reason = str(exc).strip() or exc.__class__.__name__
        # Keep the reason brief so callers can log it cleanly.
        reason = reason.splitlines()[0][:120]
        return {
            "ok": False,
            "score": None,
            "rationale": f"llm signal unavailable: {reason}",
        }


# --- Stylometric helpers -----------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s+|$)|\n+")
_WORD_RE = re.compile(r"[A-Za-z']+")
# Length >= 3 so common acronyms (AI, US, UK) don't count as shouting.
_ALL_CAPS_RE = re.compile(r"\b[A-Z]{3,}\b")
_INTERJECTIONS = {"ok", "lol", "honestly", "tbh", "yeah"}


def _split_sentences(text: str) -> list[str]:
    """Split text into non-empty sentences on [.!?]+ boundaries and newlines."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def _burstiness(text: str) -> float:
    """Sentence-length variation. Humans vary a lot; AI is metronomic.

    Measures: coefficient of variation (stdev/mean) of per-sentence word counts.
    Blind spot: very short inputs (one or two sentences) give unstable estimates.
    """
    sentences = _split_sentences(text)
    lengths = [len(_WORD_RE.findall(s)) for s in sentences]
    lengths = [n for n in lengths if n > 0]
    if len(lengths) < 2:
        # Not enough sentences to judge rhythm; treat as neutral.
        return 0.5
    mean = statistics.mean(lengths)
    if mean == 0:
        return 0.5
    cv = statistics.stdev(lengths) / mean
    # cv >= 0.8 -> 0.0 (very human); cv <= 0.15 -> 1.0 (very AI); linear between.
    if cv >= 0.8:
        return 0.0
    if cv <= 0.15:
        return 1.0
    return (0.8 - cv) / (0.8 - 0.15)


def _ttr(text: str) -> float:
    """Type-token ratio band. Clean mid-high vocabulary reads as AI-like.

    Measures: unique/total words over the first 200 words, mapped so a mid-high
    band peaks as AI-like while repetition or very rich vocabulary reads human.
    Blind spot: TTR is length-sensitive, so short texts are noisy.
    """
    words = [w.lower() for w in _WORD_RE.findall(text)][:200]
    if not words:
        return 0.5
    ttr = len(set(words)) / len(words)
    # Triangular map: peak 1.0 at ttr == 0.665, falling to 0.0 at 0.35 and 0.95.
    peak = 0.665
    if ttr <= 0.35 or ttr >= 0.95:
        return 0.0
    if ttr <= peak:
        return (ttr - 0.35) / (peak - 0.35)
    return (0.95 - ttr) / (0.95 - peak)


def _informality(text: str) -> float:
    """Informal-marker density. Casual tics signal a human writer.

    Measures: informal markers per 100 words (contractions, ellipses, "?!",
    ALL CAPS words, lowercase sentence starts, interjections like "lol"/"tbh").
    Blind spot: formal-but-human writing has few markers and can look AI-like.
    """
    words = _WORD_RE.findall(text)
    total_words = len(words)
    if total_words == 0:
        return 0.5

    markers = 0
    # Contractions: apostrophe inside a word.
    markers += len(re.findall(r"[A-Za-z]'[A-Za-z]", text))
    # Ellipses.
    markers += text.count("...")
    # Interrobangs.
    markers += text.count("?!") + text.count("!?")
    # ALL CAPS words of length >= 3 (shouting, not acronyms).
    markers += len(_ALL_CAPS_RE.findall(text))
    # Sentences starting with a lowercase letter.
    for sentence in _split_sentences(text):
        if sentence and sentence[0].isalpha() and sentence[0].islower():
            markers += 1
    # Single-word interjections.
    markers += sum(1 for w in words if w.lower() in _INTERJECTIONS)

    rate = markers / total_words * 100.0
    # rate >= 4 -> 0.0 (very human); rate == 0 -> 1.0 (clean, AI-like); linear.
    if rate >= 4:
        return 0.0
    return (4.0 - rate) / 4.0


def stylometric_signal(text: str) -> dict:
    """Combine three surface statistics into one AI-likelihood score.

    Measures: burstiness, type-token ratio, and informality, each normalized to
    [0, 1] where 1.0 = AI-like, then averaged.
    Blind spot: purely surface-level, so it can be gamed and struggles on very
    short passages.

    Returns {"score": float, "metrics": {"burstiness", "ttr", "informality"}}.
    """
    burstiness = round(_clamp(_burstiness(text)), 4)
    ttr = round(_clamp(_ttr(text)), 4)
    informality = round(_clamp(_informality(text)), 4)
    score = round((burstiness + ttr + informality) / 3.0, 4)
    return {
        "score": score,
        "metrics": {
            "burstiness": burstiness,
            "ttr": ttr,
            "informality": informality,
        },
    }


if __name__ == "__main__":
    # Offline smoke test: exercises only the stylometric path (no Groq call).
    _sample = (
        "It is important to note that the implications are significant. "
        "Furthermore, the framework provides a comprehensive foundation. "
        "Moreover, the results demonstrate a robust and scalable approach. "
        "In conclusion, the analysis underscores the value of the method."
    )
    print(stylometric_signal(_sample))
