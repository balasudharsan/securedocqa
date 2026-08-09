"""In-memory session state and the cost gate (Decisions 4, 6, 12).

All state lives in plain dicts on a single SessionStore instance — no network,
no external calls, no persistence. check_budget is the single gate that every
paid request passes through; it runs the four caps cheapest-first so nothing
reaches an LLM call once a limit is hit. Time is injectable everywhere (an
optional `now`) so tests never sleep. There is no document content here, so
nothing is logged.
"""

from __future__ import annotations

import uuid
from collections import deque
from datetime import datetime, timedelta, timezone

from app.config import settings

RATE_WINDOW_SECONDS = 60


class RateLimitExceeded(Exception):
    """Too many requests within the rolling 60-second window."""


class SessionQuestionLimitExceeded(Exception):
    """This session has reached SESSION_MAX_QUESTIONS."""


class SessionTokenLimitExceeded(Exception):
    """This session has reached SESSION_TOKEN_BUDGET."""


class DailyLimitExceeded(Exception):
    """The global daily question budget is exhausted."""


class UnknownSession(Exception):
    """No session with the given id exists in the store."""


def _now(now: datetime | None) -> datetime:
    """Resolve an injected time or default to UTC now (single source)."""
    return now if now is not None else datetime.now(timezone.utc)


class SessionStore:
    """Single in-memory store of per-session and global daily state."""

    def __init__(self, now: datetime | None = None) -> None:
        self._sessions: dict[str, dict] = {}
        self._daily_count: int = 0
        self._daily_date = _now(now).date()

    def _get(self, session_id: str) -> dict:
        """Look up session state or raise UnknownSession (never a bare KeyError)."""
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise UnknownSession from exc

    # --- lifecycle ----------------------------------------------------------

    def create_session(self, now: datetime | None = None) -> str:
        """Create state under an unguessable id (uuid4) and return the id."""
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = {
            "question_count": 0,
            "token_count": 0,
            "timestamps": deque(),
            "last_active": _now(now),
        }
        return session_id

    def touch(self, session_id: str, now: datetime | None = None) -> None:
        """Mark a session active. Called on any activity."""
        self._get(session_id)["last_active"] = _now(now)

    def delete_session(self, session_id: str) -> None:
        """Remove a session from the store (idempotent)."""
        self._sessions.pop(session_id, None)

    # --- the cost gate (Decision 12) ---------------------------------------

    def check_budget(self, session_id: str, now: datetime | None = None) -> None:
        """Run the four caps cheapest-first; raise on the first that fails.

        Order matters: rate limit, then session questions, then session tokens,
        then the daily budget — so nothing proceeds to a paid call once any cap
        is hit. Returns None if all pass.
        """
        now = _now(now)
        state = self._get(session_id)
        self._check_rate(state, now)
        self._check_session_questions(state)
        self._check_session_tokens(state)
        self._check_daily(now)

    def _check_rate(self, state: dict, now: datetime) -> None:
        cutoff = now - timedelta(seconds=RATE_WINDOW_SECONDS)
        recent = sum(1 for ts in state["timestamps"] if ts > cutoff)
        if recent >= settings.RATE_LIMIT_PER_MIN:
            raise RateLimitExceeded

    def _check_session_questions(self, state: dict) -> None:
        if state["question_count"] >= settings.SESSION_MAX_QUESTIONS:
            raise SessionQuestionLimitExceeded

    def _check_session_tokens(self, state: dict) -> None:
        if state["token_count"] >= settings.SESSION_TOKEN_BUDGET:
            raise SessionTokenLimitExceeded

    def _check_daily(self, now: datetime) -> None:
        self._roll_daily(now)
        if self._daily_count >= settings.DAILY_MAX_QUESTIONS:
            raise DailyLimitExceeded

    def _roll_daily(self, now: datetime) -> None:
        """Reset the global daily counter when the date rolls over."""
        if now.date() != self._daily_date:
            self._daily_date = now.date()
            self._daily_count = 0

    # --- usage accounting ---------------------------------------------------

    def record_usage(
        self, session_id: str, tokens_used: int, now: datetime | None = None
    ) -> None:
        """Record a successful answer's cost against session and daily counters."""
        now = _now(now)
        state = self._get(session_id)
        state["timestamps"].append(now)
        state["question_count"] += 1
        state["token_count"] += tokens_used
        self._roll_daily(now)
        self._daily_count += 1
        state["last_active"] = now

    # --- expiry (Decision 6) ------------------------------------------------

    def sweep_expired(self, now: datetime | None = None) -> list[str]:
        """Return ids whose last_active is older than the TTL (caller cleans up)."""
        now = _now(now)
        cutoff = now - timedelta(minutes=settings.SESSION_TTL_MINUTES)
        return [
            session_id
            for session_id, state in self._sessions.items()
            if state["last_active"] < cutoff
        ]
