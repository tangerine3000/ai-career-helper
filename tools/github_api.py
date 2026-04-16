"""
Fetch public GitHub repository data for a user.

Uses the unauthenticated GitHub API by default (60 req/hr).
Set GITHUB_TOKEN env var to raise the limit to 5 000 req/hr.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tools.http_get import http_get

_API_BASE = "https://api.github.com"
_MAX_REPOS = 20         # repos to inspect
_MAX_LANG_REPOS = 10    # repos to fetch language breakdowns for (extra API calls)


def fetch_github_repos(username: str) -> dict[str, Any]:
    """
    Return public, non-forked repos for *username*.

    Returns:
        {"repos": [...], "username": str}
        or {"error": str, "repos": []}
    """
    token = os.environ.get("GITHUB_TOKEN")
    auth_header = {"Authorization": f"Bearer {token}"} if token else {}

    # ── 1. Repo list ──────────────────────────────────────────────────
    list_url = (
        f"{_API_BASE}/users/{username}/repos"
        f"?sort=updated&direction=desc&per_page={_MAX_REPOS}"
    )
    list_result = http_get(list_url, extra_headers={**auth_header, "Accept": "application/vnd.github+json"})

    if "error" in list_result:
        return {"error": list_result["error"], "repos": []}

    try:
        raw_repos = json.loads(list_result["text"])
    except json.JSONDecodeError:
        return {"error": "Could not parse GitHub API response.", "repos": []}

    if isinstance(raw_repos, dict):
        # GitHub returns {"message": "..."} on errors
        msg = raw_repos.get("message", "Unknown GitHub API error")
        return {"error": f"GitHub API: {msg}", "repos": []}

    if not isinstance(raw_repos, list):
        return {"error": "Unexpected GitHub API response format.", "repos": []}

    # Filter forks; keep originals only
    own_repos = [r for r in raw_repos if not r.get("fork")][:_MAX_REPOS]

    # ── 2. Language breakdown (parallel, top repos by stars) ──────────
    top_repos = sorted(own_repos, key=lambda r: r.get("stargazers_count", 0), reverse=True)
    lang_map: dict[str, list[str]] = {}

    def _fetch_languages(repo: dict) -> tuple[str, list[str]]:
        lang_url = f"{_API_BASE}/repos/{username}/{repo['name']}/languages"
        result = http_get(lang_url, extra_headers={**auth_header, "Accept": "application/vnd.github+json"})
        if "error" in result:
            primary = repo.get("language")
            return repo["name"], [primary] if primary else []
        try:
            langs = list(json.loads(result["text"]).keys())
        except (json.JSONDecodeError, AttributeError):
            primary = repo.get("language")
            langs = [primary] if primary else []
        return repo["name"], langs

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fetch_languages, r): r for r in top_repos[:_MAX_LANG_REPOS]}
        for future in as_completed(futures):
            name, langs = future.result()
            lang_map[name] = langs

    # ── 3. Build output ───────────────────────────────────────────────
    repos = []
    for repo in own_repos:
        name = repo.get("name", "")
        primary_lang = repo.get("language")
        languages = lang_map.get(name) or ([primary_lang] if primary_lang else [])

        repos.append({
            "name": name,
            "description": repo.get("description") or "",
            "languages": languages,
            "stars": repo.get("stargazers_count", 0),
            "topics": repo.get("topics") or [],
            "url": repo.get("html_url", ""),
        })

    return {"repos": repos, "username": username}
