import time
from typing import Any

try:
    from ddgs import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    _DDGS_AVAILABLE = False


def web_search(query: str, max_results: int = 10) -> dict[str, Any]:
    """
    Search the web using DuckDuckGo.

    Returns:
        {
            "results": [{"title": str, "url": str, "snippet": str}, ...],
            "count": int
        }
        or {"error": str, "results": [], "count": 0} on failure.
    """
    if not _DDGS_AVAILABLE:
        return {
            "error": "ddgs is not installed. Run: pip install ddgs",
            "results": [],
            "count": 0,
        }

    max_results = min(max_results, 20)

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
        return {"results": results, "count": len(results)}

    except Exception as exc:
        # Brief back-off on rate-limit-style errors before returning
        if "ratelimit" in str(exc).lower() or "202" in str(exc):
            time.sleep(2)
        return {"error": str(exc), "results": [], "count": 0}
