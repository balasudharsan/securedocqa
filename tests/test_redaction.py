import logging

from app.security.redaction import log_redaction, redact


def test_email_redacted():
    redacted, counts = redact("Contact us at jane.doe@example.com please.")
    assert "[EMAIL]" in redacted
    assert "jane.doe@example.com" not in redacted
    assert counts == {"EMAIL": 1}


def test_phone_formats_redacted():
    redacted, counts = redact("Call 415-555-0132 or (415) 555 0199 today.")
    assert redacted.count("[PHONE]") == 2
    assert "415-555-0132" not in redacted
    assert counts.get("PHONE") == 2


def test_card_not_partially_matched_as_phone():
    # 16-digit card must become one [CARD], never a [PHONE] fragment.
    redacted, counts = redact("Card: 4111 1111 1111 1111 on file.")
    assert "[CARD]" in redacted
    assert "[PHONE]" not in redacted
    assert counts == {"CARD": 1}
    assert "4111" not in redacted


def test_ssn_redacted_as_id():
    redacted, counts = redact("SSN 123-45-6789 recorded.")
    assert "[ID]" in redacted
    assert "123-45-6789" not in redacted
    assert counts == {"ID": 1}


def test_aadhaar_12_digit_redacted_as_id():
    redacted, counts = redact("Aadhaar 1234 5678 9012 verified.")
    assert "[ID]" in redacted
    assert counts == {"ID": 1}


def test_multiple_pii_types_all_redacted():
    text = "Email a@b.com, call 415-555-0132, card 4111111111111111, ssn 123-45-6789."
    redacted, counts = redact(text)
    assert counts == {"CARD": 1, "ID": 1, "PHONE": 1, "EMAIL": 1}
    for placeholder in ("[EMAIL]", "[PHONE]", "[CARD]", "[ID]"):
        assert placeholder in redacted


def test_benign_text_unchanged():
    benign = "This Agreement shall be governed by the laws of the State of Delaware."
    redacted, counts = redact(benign)
    assert redacted == benign
    assert counts == {}


def test_no_original_pii_leaks_into_output():
    originals = [
        "jane.doe@example.com",
        "415-555-0132",
        "4111111111111111",
        "123-45-6789",
    ]
    text = f"{originals[0]} {originals[1]} {originals[2]} {originals[3]}"
    redacted, _ = redact(text)
    for value in originals:
        assert value not in redacted


def test_log_redaction_empty_logs_nothing(caplog):
    with caplog.at_level(logging.INFO):
        log_redaction({}, "sess-1")
        log_redaction({"EMAIL": 0}, "sess-1")
    assert caplog.records == []


def test_log_redaction_logs_counts_not_pii(caplog):
    text = "reach jane.doe@example.com or 4111111111111111"
    _, counts = redact(text)
    with caplog.at_level(logging.INFO):
        log_redaction(counts, "sess-xyz")

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.INFO
    message = record.getMessage()
    assert "pii_redacted" in message
    assert "sess-xyz" in message
    assert "EMAIL" in message and "CARD" in message
    assert "jane.doe@example.com" not in message
    assert "4111111111111111" not in message
