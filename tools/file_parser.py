"""
Parse resume / CV files (PDF, DOCX, TXT) into plain text.

Dependencies (all optional — missing ones return a clear error):
  pdfplumber  — PDF text extraction
  python-docx — DOCX extraction
"""

from pathlib import Path
from typing import Any

try:
    import pdfplumber
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False


def parse_file(file_path: str) -> dict[str, Any]:
    """
    Parse a local resume or CV file.

    Returns:
        {"text": str, "format": str, "page_count": int}
        or {"error": str} on failure.
    """
    path = Path(file_path)

    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix in (".docx", ".doc"):
        return _parse_docx(path)
    elif suffix == ".txt":
        return _parse_txt(path)
    else:
        return {
            "error": (
                f"Unsupported file format '{suffix}'. "
                "Please convert to PDF, DOCX, or TXT."
            )
        }


# ---------------------------------------------------------------------------
# Format handlers
# ---------------------------------------------------------------------------

def _parse_pdf(path: Path) -> dict[str, Any]:
    if not _PDF_AVAILABLE:
        return {
            "error": "pdfplumber is not installed. Run: pip install pdfplumber"
        }
    try:
        pages_text: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text.strip())

        if not pages_text:
            return {"error": "PDF appears to be image-only or empty (no extractable text)."}

        return {
            "text": "\n\n".join(pages_text),
            "format": "pdf",
            "page_count": len(pages_text),
        }
    except Exception as exc:
        return {"error": f"PDF parse error: {exc}"}


def _parse_docx(path: Path) -> dict[str, Any]:
    if not _DOCX_AVAILABLE:
        return {
            "error": "python-docx is not installed. Run: pip install python-docx"
        }
    try:
        doc = DocxDocument(str(path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        if not paragraphs:
            return {"error": "DOCX appears to be empty."}

        return {
            "text": "\n".join(paragraphs),
            "format": "docx",
            "page_count": None,
        }
    except Exception as exc:
        return {"error": f"DOCX parse error: {exc}"}


def _parse_txt(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return {"error": "Text file is empty."}
        return {"text": text, "format": "txt", "page_count": None}
    except Exception as exc:
        return {"error": f"TXT read error: {exc}"}
