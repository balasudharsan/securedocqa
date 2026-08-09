"""Layer 1 injection screening — DETECTION ONLY.

This module is best-effort, pattern-based *detection* for visibility and
logging. It is trivially evaded (rephrasing, spacing, encoding, translation)
and is NOT the real defense against prompt injection. The prompt boundary
(a separate module) is the actual protection. Nothing here blocks, rejects,
or modifies the document — it only flags suspicious phrasing and returns a
result for logging.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# (category_label, compiled_pattern). Case-insensitive. Deliberately small and
# readable — a representative sample, not an exhaustive filter.
INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("instruction_override", re.compile(
        r"(ignore|disregard)\s+(the\s+)?(above|previous|all|prior)\b", re.IGNORECASE)),
    ("system_prompt_exfil", re.compile(
        r"(reveal|print|show|repeat)\s+(me\s+)?(your\s+)?(system\s+)?(prompt|instructions)"
        r"|what\s+are\s+your\s+instructions", re.IGNORECASE)),
    ("role_hijack", re.compile(
        r"you\s+are\s+now\b|\bact\s+as\b|pretend\s+(you\s+are|to\s+be)", re.IGNORECASE)),
    ("delimiter_injection", re.compile(
        r"^\s*(system|assistant)\s*:", re.IGNORECASE | re.MULTILINE)),
]


def screen_text(text: str) -> list[str]:
    """Return category labels for any injection patterns found (empty = none).

    Pure detection: never raises, never modifies the text. A match means
    "worth logging", not "block this" — see the module docstring.
    """
    return [label for label, pattern in INJECTION_PATTERNS if pattern.search(text)]


def is_suspicious(text: str) -> bool:
    """True if screen_text flagged at least one category. Thin wrapper."""
    return bool(screen_text(text))


def log_screening(categories: list[str], session_id: str) -> None:
    """Log one structured WARNING when categories were flagged; else nothing.

    Logs ONLY the category labels and session id — never the document text or
    the matched substring. The screened text is attacker-controlled and may be
    sensitive, so it must never enter the logs.
    """
    if not categories:
        return
    logger.warning(
        "event=injection_detected categories=%s session_id=%s",
        categories,
        session_id,
    )
