import io

import pytest
from pypdf import PdfWriter

from app.config import settings
from app.pipeline.extract import (
    FileTooLargeError,
    InvalidFileTypeError,
    NoTextFoundError,
    assert_has_text,
    validate_file,
)


def _tiny_pdf_bytes() -> bytes:
    """A minimal, real PDF (one blank page) generated in-memory."""
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_validate_file_rejects_non_pdf_filename():
    with pytest.raises(InvalidFileTypeError):
        validate_file(b"%PDF-1.4 ...", "resume.docx")


def test_validate_file_rejects_missing_magic_number():
    # .pdf extension but not actually a PDF — spoofed extension.
    with pytest.raises(InvalidFileTypeError):
        validate_file(b"not a real pdf", "evil.pdf")


def test_validate_file_rejects_oversized_bytes():
    oversized = b"%PDF-" + b"0" * (settings.MAX_FILE_MB * 1024 * 1024)
    with pytest.raises(FileTooLargeError):
        validate_file(oversized, "big.pdf")


def test_validate_file_passes_for_valid_pdf():
    assert validate_file(_tiny_pdf_bytes(), "doc.PDF") is None


def test_assert_has_text_raises_for_near_empty_pages():
    with pytest.raises(NoTextFoundError):
        assert_has_text([(1, "short"), (2, "")])


def test_assert_has_text_passes_with_real_text():
    pages = [(1, "x" * 60), (2, "y" * 60)]
    assert assert_has_text(pages) is None
