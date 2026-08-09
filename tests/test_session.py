from datetime import datetime, timedelta, timezone

import pytest

from app.config import settings
from app.session.state import (
    DailyLimitExceeded,
    RateLimitExceeded,
    SessionQuestionLimitExceeded,
    SessionStore,
    SessionTokenLimitExceeded,
    UnknownSession,
)

T0 = datetime(2026, 8, 6, 12, 0, 0, tzinfo=timezone.utc)


def test_create_session_unique_and_initialised():
    store = SessionStore(now=T0)
    a = store.create_session(now=T0)
    b = store.create_session(now=T0)
    assert a != b
    state = store._sessions[a]
    assert state["question_count"] == 0
    assert state["token_count"] == 0
    assert list(state["timestamps"]) == []
    assert state["last_active"] == T0


def test_check_budget_passes_on_fresh_session():
    store = SessionStore(now=T0)
    sid = store.create_session(now=T0)
    assert store.check_budget(sid, now=T0) is None


def test_rate_limit_raises_after_limit_within_window():
    store = SessionStore(now=T0)
    sid = store.create_session(now=T0)
    for i in range(settings.RATE_LIMIT_PER_MIN):
        store.record_usage(sid, tokens_used=1, now=T0 + timedelta(seconds=i))
    with pytest.raises(RateLimitExceeded):
        store.check_budget(sid, now=T0 + timedelta(seconds=settings.RATE_LIMIT_PER_MIN))


def test_rate_limit_ignores_timestamps_older_than_window():
    store = SessionStore(now=T0)
    sid = store.create_session(now=T0)
    for _ in range(settings.RATE_LIMIT_PER_MIN):
        store.record_usage(sid, tokens_used=1, now=T0)
    # 61s later all prior timestamps have aged out of the 60s window.
    assert store.check_budget(sid, now=T0 + timedelta(seconds=61)) is None


def test_session_question_cap():
    store = SessionStore(now=T0)
    sid = store.create_session(now=T0)
    store._sessions[sid]["question_count"] = settings.SESSION_MAX_QUESTIONS
    with pytest.raises(SessionQuestionLimitExceeded):
        store.check_budget(sid, now=T0 + timedelta(minutes=5))


def test_session_token_cap():
    store = SessionStore(now=T0)
    sid = store.create_session(now=T0)
    store._sessions[sid]["token_count"] = settings.SESSION_TOKEN_BUDGET
    with pytest.raises(SessionTokenLimitExceeded):
        store.check_budget(sid, now=T0 + timedelta(minutes=5))


def test_daily_cap_raises():
    store = SessionStore(now=T0)
    sid = store.create_session(now=T0)
    store._daily_count = settings.DAILY_MAX_QUESTIONS
    with pytest.raises(DailyLimitExceeded):
        store.check_budget(sid, now=T0 + timedelta(minutes=5))


def test_daily_cap_resets_on_new_day():
    store = SessionStore(now=T0)
    sid = store.create_session(now=T0)
    store._daily_count = settings.DAILY_MAX_QUESTIONS
    next_day = T0 + timedelta(days=1)
    # Rolling into a new date resets the daily counter, so the gate passes.
    assert store.check_budget(sid, now=next_day) is None
    assert store._daily_count == 0


def test_record_usage_increments_counters():
    store = SessionStore(now=T0)
    sid = store.create_session(now=T0)
    store.record_usage(sid, tokens_used=120, now=T0)
    state = store._sessions[sid]
    assert state["question_count"] == 1
    assert state["token_count"] == 120
    assert len(state["timestamps"]) == 1
    assert state["last_active"] == T0
    assert store._daily_count == 1


def test_sweep_expired_returns_only_stale_sessions():
    store = SessionStore(now=T0)
    fresh = store.create_session(now=T0)
    stale = store.create_session(now=T0)
    # Age only the stale one past the TTL.
    store._sessions[stale]["last_active"] = T0 - timedelta(
        minutes=settings.SESSION_TTL_MINUTES + 1)
    expired = store.sweep_expired(now=T0)
    assert stale in expired
    assert fresh not in expired


def test_unknown_session_raises_on_lookup_methods():
    store = SessionStore(now=T0)
    for call in (
        lambda: store.check_budget("nope", now=T0),
        lambda: store.record_usage("nope", tokens_used=1, now=T0),
        lambda: store.touch("nope", now=T0),
    ):
        with pytest.raises(UnknownSession):
            call()


def test_delete_unknown_session_does_not_raise():
    store = SessionStore(now=T0)
    # Deleting an id that isn't present is a no-op, not an error.
    assert store.delete_session("nope") is None


def test_cheapest_cap_raises_first_when_many_exceeded():
    store = SessionStore(now=T0)
    sid = store.create_session(now=T0)
    # Exceed rate, session-questions, tokens, and daily all at once...
    for _ in range(settings.SESSION_MAX_QUESTIONS):
        store.record_usage(sid, tokens_used=settings.SESSION_TOKEN_BUDGET, now=T0)
    store._daily_count = settings.DAILY_MAX_QUESTIONS
    # ...the cheapest check (rate limit) must be the one that raises.
    with pytest.raises(RateLimitExceeded):
        store.check_budget(sid, now=T0)
