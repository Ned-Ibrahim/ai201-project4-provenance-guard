"""Provenance Guard: score fusion and user-facing labels.

Combines the LLM and stylometric signals into a single confidence, maps that
confidence to an attribution bucket, and provides the exact copy shown to users.
"""

from __future__ import annotations

from typing import Optional

# Confidence thresholds for attribution buckets.
CONF_LIKELY_AI = 0.75
CONF_LIKELY_HUMAN = 0.40


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a numeric value into the inclusive [low, high] range."""
    return max(low, min(high, value))


def combine_signals(llm_score: Optional[float], stylo_score: float) -> float:
    """Fuse the two signals into one AI-likelihood confidence in [0, 1].

    Measures: a weighted blend that leans on the LLM (0.65) over stylometry
    (0.35), pulling toward 0.5 when the two signals strongly disagree.
    Blind spot: when the LLM signal is unavailable it falls back to a hedged
    stylometry-only band ([0.35, 0.65]), deliberately refusing to be confident.

    Returns the confidence rounded to 4 decimal places and clamped to [0, 1].
    """
    if llm_score is None:
        # LLM degraded: trust stylometry only, but never strongly.
        confidence = _clamp(stylo_score, 0.35, 0.65)
        return round(_clamp(confidence), 4)

    raw = 0.65 * llm_score + 0.35 * stylo_score
    disagreement = abs(llm_score - stylo_score)
    if disagreement > 0.35:
        # Pull the confidence toward 0.5 in proportion to the disagreement.
        confidence = raw + (0.5 - raw) * min(1.0, (disagreement - 0.35) / 0.45)
    else:
        confidence = raw
    return round(_clamp(confidence), 4)


def attribution_for(confidence: float) -> str:
    """Map a confidence score to an attribution bucket.

    Measures: which side of the two thresholds the confidence falls on.
    Blind spot: a hard threshold means a hair's-width difference can flip the
    label, so the "uncertain" band exists to soften borderline calls.
    """
    if confidence >= CONF_LIKELY_AI:
        return "likely_ai"
    if confidence <= CONF_LIKELY_HUMAN:
        return "likely_human"
    return "uncertain"


LABELS: dict[str, str] = {
    "likely_ai": (
        "**Likely AI-generated.** Our automated analysis found strong signs "
        "that this piece was written by an AI tool rather than a person. This "
        "is an automated assessment and can be wrong. If you are the creator "
        "and you wrote this yourself, you can appeal this label and a person "
        "will review it."
    ),
    "uncertain": (
        "**Origin unclear.** Our automated analysis couldn't confidently "
        "determine whether this piece was written by a person or by an AI "
        "tool. Some signals point each way. Please read with your own "
        "judgment: this label reflects genuine uncertainty, not an accusation."
    ),
    "likely_human": (
        "**Likely human-written.** Our automated analysis found strong signs "
        "that this piece was written by a person. No automated check is "
        "perfect, but nothing here suggests AI generation."
    ),
}


def label_for(attribution: str) -> str:
    """Return the exact user-facing label text for an attribution bucket.

    Measures: nothing statistical; it is a verbatim lookup into LABELS.
    Blind spot: an unknown attribution key has no copy and will raise KeyError.
    """
    return LABELS[attribution]
