"""PDF validation and text extraction.

This is the app's highest-risk surface: it parses attacker-controlled PDF
bytes. Every guard here exists to fail fast and fail typed, so callers never
have to catch a raw pypdf exception or trust a filename.
"""

from __future__ import annotations

import io

from pypdf import PdfReader

from app.config import settings

PDF_MAGIC = b"%PDF-"
MIN_TEXT_CHARS = 100


class FileTooLargeError(Exception):
    """Uploaded file exceeds settings.MAX_FILE_MB."""


class TooManyPagesError(Exception):
    """PDF has more pages than settings.MAX_PAGES."""


class InvalidFileTypeError(Exception):
    """File is not a PDF by extension and/or magic number."""


class UnreadablePDFError(Exception):
    """pypdf could not parse the bytes (corrupt/encrypted/malformed)."""


class NoTextFoundError(Exception):
    """No extractable text — likely a scanned/image-only PDF."""


def validate_file(file_bytes: bytes, filename: str) -> None:
    """Reject non-PDFs and oversized files BEFORE any parsing.

    Checks both extension and magic number: an extension alone is trivially
    spoofed, so we also require the bytes to begin with %PDF-. Size is
    checked here (pre-parse) so a huge upload can never reach pypdf.
    """
    if not filename.lower().endswith(".pdf") or not file_bytes.startswith(PDF_MAGIC):
        raise InvalidFileTypeError("File must be a .pdf whose bytes start with %PDF-.")
    if len(file_bytes) > settings.MAX_FILE_MB * 1024 * 1024:
        raise FileTooLargeError(f"File exceeds {settings.MAX_FILE_MB} MB limit.")


def extract_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    """Parse a PDF into (page_number, text) pairs, page numbers 1-based.

    All pypdf usage is wrapped: a corrupt or malicious PDF raises our typed
    UnreadablePDFError, never a raw library exception. Page count is checked
    before extracting any text so huge documents fail fast.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        page_count = len(reader.pages)
    except Exception as exc:
        raise UnreadablePDFError("Could not read PDF.") from exc

    if page_count > settings.MAX_PAGES:
        raise TooManyPagesError(f"PDF has {page_count} pages; limit is {settings.MAX_PAGES}.")

    try:
        return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]
    except Exception as exc:
        raise UnreadablePDFError("Could not extract text from PDF.") from exc


def assert_has_text(pages: list[tuple[int, str]]) -> None:
    """Reject PDFs with essentially no extractable text.

    Under MIN_TEXT_CHARS total signals a scanned/image-only PDF that our
    text pipeline cannot use — surface it as a typed error rather than
    silently indexing empty content.
    """
    total = sum(len(text) for _, text in pages)
    if total < MIN_TEXT_CHARS:
        raise NoTextFoundError("No extractable text found (scanned or image-only PDF?).")
