"""
normalizer.py
─────────────
Text normalization BEFORE any classification.
Fixes encoding artifacts, strips HTML, handles Bahasa/English mixed input.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


# ── Malay / EN common contractions & informal spellings ──────────────────────
_INFORMAL_MAP = {
    r"\btak\b": "tidak",
    r"\bxde\b": "tiada",
    r"\bxbole\b": "tidak boleh",
    r"\bokay\b": "ok",
    r"\bu\b": "you",
    r"\bbtw\b": "by the way",
    r"\bfyi\b": "for your information",
    r"\btq\b": "thank you",
    r"\bthx\b": "thank you",
}


@dataclass
class NormalizationResult:
    original: str
    normalized: str
    changes: list[str]


def normalize_text(text: str) -> NormalizationResult:
    """
    Full normalization pipeline:
      1. Unicode NFKC  → fix mojibake / half-width chars
      2. Strip HTML tags
      3. Decode HTML entities
      4. Remove zero-width / invisible chars
      5. Normalize whitespace
      6. Lowercase
      7. Map informal spellings (MY context)
      8. Strip punctuation clusters (keep sentence structure)
    """
    changes: list[str] = []
    original = text

    # 1. Unicode normalization
    step = unicodedata.normalize("NFKC", text)
    if step != text:
        changes.append("unicode_normalized")
    text = step

    # 2. Strip HTML tags
    step = re.sub(r"<[^>]+>", " ", text)
    if step != text:
        changes.append("html_stripped")
    text = step

    # 3. HTML entities (basic set; add more as needed)
    entity_map = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&nbsp;": " ", "&quot;": '"', "&#39;": "'",
    }
    for entity, char in entity_map.items():
        text = text.replace(entity, char)

    # 4. Remove zero-width / invisible Unicode chars
    step = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
    if step != text:
        changes.append("invisible_chars_removed")
    text = step

    # 5. Collapse whitespace
    step = re.sub(r"\s+", " ", text).strip()
    if step != text:
        changes.append("whitespace_normalized")
    text = step

    # 6. Lowercase
    text = text.lower()
    changes.append("lowercased")

    # 7. Informal MY/EN spelling normalization
    for pattern, replacement in _INFORMAL_MAP.items():
        new = re.sub(pattern, replacement, text)
        if new != text:
            changes.append(f"informal_mapped:{pattern}")
        text = new

    # 8. Collapse repeated punctuation  e.g. "!!!" → "!"
    step = re.sub(r"([!?.,])\1+", r"\1", text)
    if step != text:
        changes.append("punct_collapsed")
    text = step

    return NormalizationResult(original=original, normalized=text, changes=changes)


def normalize_conversation_messages(messages: list[dict]) -> list[dict]:
    """Normalize the text content of every message in a conversation."""
    normalized = []
    for msg in messages:
        m = dict(msg)
        if "text" in m and isinstance(m["text"], str):
            result = normalize_text(m["text"])
            m["text_normalized"] = result.normalized
            m["normalization_changes"] = result.changes
        normalized.append(m)
    return normalized
