"""Response parser for ET1 experiments.

Extracts (answer, confidence) from raw model outputs regardless of format.
Uses a fallback chain: JSON → stripped markdown JSON → regex JSON → text.

The parser is format-agnostic by design: a model trained on C1 (plain text)
and a model trained on C4 (jsonld-ex) will produce very different outputs,
and the parser must handle both.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Verbal confidence mapping (from protocol §7.2 evaluation config)
# ---------------------------------------------------------------------------

_VERBAL_CONFIDENCE: list[tuple[str, float]] = [
    # Sorted longest-first to match multi-word phrases before single words
    ("don't have enough information", 0.05),
    ("cannot be confirmed", 0.05),
    ("not enough information", 0.05),
    ("don't know", 0.0),
    ("no idea", 0.1),
    ("very uncertain", 0.2),
    ("very confident", 0.9),
    ("fairly confident", 0.8),
    ("somewhat confident", 0.65),
    ("not sure", 0.4),
    ("not confident", 0.3),
    ("uncertain", 0.5),
    ("doubtful", 0.3),
    ("confident", 0.8),
    ("certain", 1.0),
    ("i'm sure", 0.9),
    ("might be", 0.4),
    ("maybe", 0.4),
    ("probably", 0.7),
    ("likely", 0.7),
    ("possibly", 0.4),
    ("perhaps", 0.4),
]

# Abstention phrases
_ABSTENTION_PHRASES = [
    "don't have enough information",
    "cannot be confirmed or denied",
    "not enough information",
    "i don't know",
    "unable to determine",
    "cannot answer",
    "no answer",
    "unanswerable",
]

# JSON metadata keys to skip when looking for the "answer" value
_METADATA_KEYS = {
    "@context", "@type", "@id", "@opinion", "@source", "@method",
    "@valid_from", "@valid_until", "name", "entity", "relation",
    "confidence", "base_rate",
}


# ---------------------------------------------------------------------------
# ParseResult
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """Result of parsing a model response."""

    answer: Optional[str]
    confidence: Optional[float]
    is_abstention: bool
    is_parseable: bool
    raw: str

    def __repr__(self) -> str:
        conf_str = f"{self.confidence:.3f}" if self.confidence is not None else "None"
        ans_str = (self.answer[:40] + "...") if self.answer and len(self.answer) > 40 else self.answer
        return (
            f"ParseResult(answer={ans_str!r}, confidence={conf_str}, "
            f"abstention={self.is_abstention}, parseable={self.is_parseable})"
        )


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _try_parse_json(raw: str) -> Optional[dict]:
    """Try to parse raw string as JSON. Returns None on failure."""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _strip_markdown_fences(raw: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers."""
    stripped = raw.strip()
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?\s*```$"
    match = re.search(pattern, stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def _find_json_in_text(raw: str) -> Optional[dict]:
    """Try to find a JSON object embedded in text."""
    # Look for {...} patterns
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
    if match:
        return _try_parse_json(match.group(0))
    return None


def _extract_answer_from_json(obj: dict) -> Optional[str]:
    """Extract the most likely 'answer' value from a parsed JSON object."""
    # Priority order for answer fields
    for key in ("value", "text", "answer"):
        if key in obj and isinstance(obj[key], str):
            return obj[key]

    # Fall back to first non-metadata string value
    for key, val in obj.items():
        if key not in _METADATA_KEYS and isinstance(val, str) and len(val) > 0:
            return val

    # Check nested objects
    for key, val in obj.items():
        if key not in _METADATA_KEYS and isinstance(val, dict):
            nested = _extract_answer_from_json(val)
            if nested:
                return nested

    return None


def _extract_confidence_from_json(obj: dict) -> Optional[float]:
    """Extract confidence from JSON: @opinion (projected) or scalar field."""
    # Check for @opinion (jsonld-ex style) — search recursively
    opinion = _find_nested(obj, "@opinion")
    if opinion and isinstance(opinion, dict):
        b = opinion.get("belief", 0)
        u = opinion.get("uncertainty", 0)
        a = opinion.get("base_rate", 0.5)
        return _clamp(b + a * u)

    # Check for scalar confidence field
    if "confidence" in obj:
        val = obj["confidence"]
        if isinstance(val, (int, float)):
            return _clamp(float(val))

    return None


def _find_nested(obj, key):
    """Recursively find a key in nested dict/list."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_nested(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_nested(item, key)
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_numeric_confidence(text: str) -> Optional[float]:
    """Extract a numeric confidence value from text like 'Confidence: 0.85'."""
    patterns = [
        r"[Cc]onfidence[:\s]+([01]?\.\d+)",
        r"[Cc]onfidence[:\s]+(1\.0|0)",
        r"(?:score|probability|certainty)[:\s]+([01]?\.\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _clamp(float(match.group(1)))
    return None


def _extract_verbal_confidence(text: str) -> Optional[float]:
    """Extract confidence from verbal expressions in text."""
    text_lower = text.lower()
    for phrase, score in _VERBAL_CONFIDENCE:
        if phrase in text_lower:
            return score
    return None


def _detect_abstention(text: str) -> bool:
    """Check if the response is an explicit abstention."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in _ABSTENTION_PHRASES)


def _clamp(value: float) -> float:
    """Clamp a float to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_response(raw: str) -> ParseResult:
    """Parse a raw model response to extract answer and confidence.

    Fallback chain:
    1. Try JSON parse directly
    2. Strip markdown fences, try JSON again
    3. Search for embedded JSON in text
    4. Fall back to plain text extraction

    Args:
        raw: The raw model output string.

    Returns:
        ParseResult with extracted answer, confidence, and metadata.
    """
    stripped = raw.strip()

    # Handle empty/whitespace input
    if not stripped:
        return ParseResult(
            answer=None, confidence=None,
            is_abstention=False, is_parseable=False, raw=raw,
        )

    # Check for abstention first (before any parsing)
    is_abstention = _detect_abstention(stripped)

    answer: Optional[str] = None
    confidence: Optional[float] = None
    parsed_json = False

    # --- Attempt 1: direct JSON parse ---
    obj = _try_parse_json(stripped)

    # --- Attempt 2: strip markdown fences ---
    if obj is None:
        cleaned = _strip_markdown_fences(stripped)
        obj = _try_parse_json(cleaned)

    # --- Attempt 3: find JSON embedded in text ---
    if obj is None:
        obj = _find_json_in_text(stripped)

    # --- If we got JSON, extract from it ---
    if obj is not None:
        parsed_json = True
        answer = _extract_answer_from_json(obj)
        confidence = _extract_confidence_from_json(obj)

    # --- Attempt 4: text-based extraction ---
    if answer is None:
        # Use the whole text as the answer (evaluator will compare)
        answer = stripped
        parsed_json = False

    # --- Extract confidence from text if not found in JSON ---
    if confidence is None:
        confidence = _extract_numeric_confidence(stripped)
    if confidence is None:
        confidence = _extract_verbal_confidence(stripped)

    # Abstention overrides confidence
    if is_abstention:
        if confidence is None:
            confidence = 0.05

    return ParseResult(
        answer=answer,
        confidence=confidence,
        is_abstention=is_abstention,
        is_parseable=True,
        raw=raw,
    )
