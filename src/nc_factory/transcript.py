from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Sequence

from num2words import num2words

COMPARISON_NORMALIZATION_VERSION = "nc-comparison-v2"
_EN_GROUPED_INTEGER = re.compile(r"(?<![\w.,:/-])-?[1-9]\d{0,2}(?:,\d{3})+(?![\d,:/-]|\.\d)")
_UNIT_ALIASES = {
    "en": {
        "metres": "meters",
        "metre": "meter",
        "kilogrammes": "kilogram",
        "kilogramme": "kilogram",
        "kilograms": "kilogram",
        "kg": "kilogram",
    },
    "sv": {"kg": "kilogram"},
}

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_for_comparison(text: str, language: str) -> str:
    """Return comparison tokens without guessing ambiguous dates or decimals.

    Uses the shared number normalizer. Invalid source-style content (ValueError)
    falls back to punctuation cleanup so ASR hallucinations remain comparable.
    """
    language = (language or "en").lower()
    if language not in ("en", "sv"):
        language = "en"
    text = unicodedata.normalize("NFC", text or "").casefold()
    if language == "en":
        # ASR commonly formats spoken thousands as 3,619. Do not reinterpret
        # decimals, dates, malformed groups or Swedish decimal commas.
        text = _EN_GROUPED_INTEGER.sub(
            lambda match: num2words(int(match.group().replace(",", "")), lang="en"), text
        )
    try:
        from .text import normalize_text

        normalized = normalize_text(text, language=language)
    except ValueError:
        normalized = unicodedata.normalize("NFC", text).casefold()
    aliases = _UNIT_ALIASES[language]
    words = [aliases.get(w, w) for w in _WORD_RE.findall(normalized.casefold()) if w != "_"]
    return " ".join(words)


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance over sequences in O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def _has_unexpected_repetition(reference_words: Sequence[str], hypothesis_words: Sequence[str]) -> bool:
    def trigrams(words: Sequence[str]) -> Counter:
        return Counter(tuple(words[i : i + 3]) for i in range(max(0, len(words) - 2)))

    ref_counts = trigrams(reference_words)
    hyp_counts = trigrams(hypothesis_words)
    for gram, count in hyp_counts.items():
        if count >= 2 and ref_counts.get(gram, 0) < count:
            return True
    return False


def transcript_metrics(reference: str, hypothesis: str, language: str = "en") -> dict:
    norm_ref = normalize_for_comparison(reference, language)
    norm_hyp = normalize_for_comparison(hypothesis, language)
    ref_words = norm_ref.split() if norm_ref else []
    hyp_words = norm_hyp.split() if norm_hyp else []

    distance = edit_distance(ref_words, hyp_words)
    denom = len(ref_words)
    wer = 0.0 if denom == 0 and not hyp_words else (distance / denom if denom else 1.0)

    ref_chars = list(norm_ref)
    hyp_chars = list(norm_hyp)
    edit_chars = edit_distance(ref_chars, hyp_chars)
    char_denom = len(ref_chars)
    cer = 0.0 if char_denom == 0 and not hyp_chars else (edit_chars / char_denom if char_denom else 1.0)

    hyp_counter = Counter(hyp_words)
    ref_counter = Counter(ref_words)
    missing_words = list((ref_counter - hyp_counter).elements())

    return {
        "wer": wer,
        "cer": cer,
        "normalized_reference": norm_ref,
        "normalized_hypothesis": norm_hyp,
        "missing_words": missing_words,
        "repeated_phrase": _has_unexpected_repetition(ref_words, hyp_words),
    }
