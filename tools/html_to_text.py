"""
Convert raw HTML to clean plain text for LLM consumption.

Uses BeautifulSoup4 when available; falls back to a simple regex strip.
"""

import re

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

# Tags whose entire subtree should be dropped (noise, not content)
_DROP_TAGS = {"script", "style", "noscript", "header", "footer", "nav", "aside"}

# Minimum text length to consider a page successfully rendered
JS_RENDER_THRESHOLD = 300


def html_to_text(html: str) -> str:
    """
    Return clean plain text from an HTML string.
    Collapses whitespace and removes boilerplate tags.
    """
    if not html:
        return ""

    if _BS4_AVAILABLE:
        return _bs4_extract(html)
    return _regex_extract(html)


def is_js_rendered(text: str) -> bool:
    """
    Return True if the page looks like it requires JavaScript to render
    (i.e. the extracted text is too short or contains JS-wall phrases).
    """
    if len(text.strip()) < JS_RENDER_THRESHOLD:
        return True
    js_wall_phrases = [
        "please enable javascript",
        "javascript is required",
        "you need to enable javascript",
        "loading...",
        "checking your browser",
    ]
    lower = text.lower()
    return any(phrase in lower for phrase in js_wall_phrases)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _bs4_extract(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()

    # Prefer the main content area if present
    main = soup.find("main") or soup.find(id="main") or soup.find(class_="job-description")
    root = main if main else soup

    text = root.get_text(separator="\n")
    return _collapse_whitespace(text)


def _regex_extract(html: str) -> str:
    # Drop script/style blocks
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Strip all remaining tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Decode common HTML entities
    text = (
        text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&nbsp;", " ")
            .replace("&#39;", "'")
            .replace("&quot;", '"')
    )
    return _collapse_whitespace(text)


def _collapse_whitespace(text: str) -> str:
    # Collapse runs of blank lines to at most one blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
