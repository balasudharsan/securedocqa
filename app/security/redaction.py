"""Gate-side PII redaction (Decision 8).

Redacts STRUCTURED PII only — emails, phone numbers, credit-card numbers, and
government IDs (US SSN xxx-xx-xxxx and 12-digit Aadhaar). Each match becomes a
TYPED placeholder ([EMAIL], [PHONE], [CARD], [ID]) so the sentence stays
readable.

NAMES ARE NOT REDACTED in v1: reliable name detection needs NER, which is
deferred to v2. This runs at the gate before text leaves to a third party, so
it is best-effort — it must never leak an original value into its output or
its logs.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ORDERED — do not reorder. Longer digit-runs MUST match first so a shorter
# pattern can never eat part of a longer number: CARD (13-19 digits) and ID
# (9/12 digits) are consumed before PHONE (10 digits) runs. Phone/number
# patterns are inherently prone to false positives; per Decision 8 this is
# accepted (best-effort, over-redaction preferred to leaking).
REDACTION_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    # 13-19 digits, optional single space/hyphen between digits (card groups).
    ("CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), "[CARD]"),
    # US SSN (xxx-xx-xxxx) OR 12-digit Aadhaar (optional space every 4 digits).
    ("ID", re.compile(r"\b(?:\d{3}-\d{2}-\d{4}|\d{4}\s?\d{4}\s?\d{4})\b"), "[ID]"),
    # 10-digit US phone in common formats, optional +1 country code. Uses digit
    # lookarounds (not \b) so a leading "(" in "(415) 555 0199" still anchors.
    ("PHONE", re.compile(
        r"(?<!\d)(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\d)"), "[PHONE]"),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
]


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Replace structured PII with typed placeholders, in pattern order.

    Returns (redacted_text, counts) where counts maps label -> number of
    replacements (labels with zero matches are omitted). Pure and
    deterministic: no I/O, no network, no logging — so it can never leak the
    original values.
    """
    counts: dict[str, int] = {}
    for label, pattern, placeholder in REDACTION_PATTERNS:
        text, n = pattern.subn(placeholder, text)
        if n:
            counts[label] = n
    return text, counts


def log_redaction(counts: dict[str, int], session_id: str) -> None:
    """Log one structured INFO when PII was redacted; otherwise nothing.

    Logs ONLY the counts and session id — never the original text or any
    matched value. By design this function is not even given the raw text.
    """
    if not any(counts.values()):
        return
    logger.info(
        "event=pii_redacted counts=%s session_id=%s",
        counts,
        session_id,
    )
