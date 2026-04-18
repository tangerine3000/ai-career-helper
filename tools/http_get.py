from typing import Any

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}
_MAX_BODY_CHARS = 500_000  # increased to handle large GitHub API responses (~100KB)


def http_get(url: str, timeout: int = 15, extra_headers: dict[str, str] | None = None) -> dict[str, Any]:
    """
    Fetch the text content of a URL.

    Returns:
        {"url": str, "status_code": int, "text": str, "content_type": str}
        or {"error": str, "url": str} on failure.
    """
    if not _HTTPX_AVAILABLE:
        return {"error": "httpx is not installed. Run: pip install httpx", "url": url}

    try:
        headers = {**_HEADERS, **(extra_headers or {})}
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()
            return {
                "url": str(response.url),
                "status_code": response.status_code,
                "text": response.text[:_MAX_BODY_CHARS],
                "content_type": response.headers.get("content-type", ""),
            }
    except httpx.TimeoutException:
        return {"error": "Request timed out", "url": url}
    except httpx.HTTPStatusError as exc:
        return {"error": f"HTTP {exc.response.status_code}", "url": url}
    except Exception as exc:
        return {"error": str(exc), "url": url}
