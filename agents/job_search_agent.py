"""
Agent 1 — Job Search Agent

Searches multiple job boards for current listings matching a given job title,
deduplicates results, and returns a list of JobListing objects.

Uses Claude (Haiku) with tool use in an agentic loop. Claude decides the
search strategy; results are submitted via a structured `submit_results` tool
call so they arrive as typed data rather than parsed text.
"""

import hashlib
import json
import os
import time
from datetime import datetime
from typing import Any, Optional

import anthropic

from models.types import JobListing, JobSearchInput
from tools.http_get import http_get
from tools.web_search import web_search

# ---------------------------------------------------------------------------
# Tool definitions passed to Claude
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": (
            "Search the web for job listings. Use targeted queries with site: operators "
            "to hit specific job boards. Returns titles, URLs, and snippets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Examples:\n"
                        '  site:linkedin.com/jobs "Senior Backend Engineer" remote\n'
                        '  site:greenhouse.io "Data Scientist" New York\n'
                        '  site:lever.co "Product Manager"\n'
                        '  "Software Engineer" jobs site:indeed.com remote'
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (1-8). Default 5.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch the text content of a URL. Use this to get additional details "
            "(company name, exact location, posted date) from a job listing page "
            "when search snippets are insufficient."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."}
            },
            "required": ["url"],
        },
    },
    {
        "name": "submit_results",
        "description": (
            "Call this when you have gathered enough job listings. "
            "Submit all discovered listings as a structured array. "
            "Do NOT call this until you have searched at least 3 different job boards."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "listings": {
                    "type": "array",
                    "description": "Array of job listings found.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url":         {"type": "string"},
                            "title":       {"type": "string"},
                            "company":     {"type": "string"},
                            "location":    {"type": "string"},
                            "posted_date": {
                                "type": "string",
                                "description": "ISO 8601 date if known, else empty string.",
                            },
                        },
                        "required": ["url", "title", "company", "location"],
                    },
                }
            },
            "required": ["listings"],
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt (cached — stable across runs)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a job search agent. Your task is to find current, \
relevant job postings across multiple job boards for a given job title and location.

## Search strategy
1. Run targeted searches across at least these sources:
   - LinkedIn Jobs  (site:linkedin.com/jobs)
   - Indeed         (site:indeed.com)
   - Greenhouse ATS (site:greenhouse.io)
   - Lever ATS      (site:lever.co)
   - Glassdoor      (site:glassdoor.com)
2. If a search returns fewer than 3 results, try an alternate phrasing or synonym.
3. If the total result count is below 5 after all searches, broaden the location or \
remove seniority qualifiers (e.g. drop "Senior").
4. Fetch individual listing URLs only when the snippet lacks company name or location.

## Deduplication
- Skip a listing if its URL is already in your collected set.
- Skip a listing if an identical (title, company) pair already exists.

## When to stop
Call `submit_results` once you have collected up to the requested max_results \
listings, or once you have exhausted reasonable search queries, whichever comes first.

## Rules
- Only include real URLs you found via search or fetch — never fabricate URLs.
- Use today's date (ISO 8601) as posted_date when it is unavailable.
"""

_RATE_LIMIT_RETRIES = 3
_RETRY_BASE_SECONDS = 2
_MAX_SEARCH_RESULTS_FOR_CONTEXT = 5
_MAX_SEARCH_SNIPPET_CHARS = 240
_MAX_FETCH_TEXT_CHARS_FOR_CONTEXT = 1_500
_MAX_MODEL_TOKENS = 1600
_MAX_AGENT_STEPS = 6
_MAX_CONTEXT_TURNS = 3
_MAX_SEARCH_RESULTS_PER_TOOL_CALL = 8
_INTER_STEP_DELAY_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class JobSearchAgent:
    """
    Agent 1 — Job Search Agent.

    Usage:
        agent = JobSearchAgent()
        listings = agent.search(JobSearchInput(job_title="Senior Backend Engineer", location="Remote"))
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        # Haiku: fast and cheap for search workload
        self.model = "claude-haiku-4-5-20251001"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, input: JobSearchInput) -> list[JobListing]:
        """
        Run the agentic search loop and return deduplicated JobListing objects.
        """
        print(
            f"[JobSearchAgent] Starting search for '{input.job_title}' "
            f"(location={input.location or 'any'}, max={input.max_results})"
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": self._build_prompt(input)}
        ]
        step = 0

        while True:
            if step >= _MAX_AGENT_STEPS:
                print(
                    f"[JobSearchAgent] Reached max steps ({_MAX_AGENT_STEPS}) "
                    "without submit_results; returning no results."
                )
                return []

            step += 1
            if step > 1:
                # Smooth request bursts in fast tool-use loops.
                time.sleep(_INTER_STEP_DELAY_SECONDS)
            print(f"[JobSearchAgent] Step {step}: requesting next action from model...")
            response = self._create_message_with_retry(messages, step)

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})
            print(f"[JobSearchAgent] Step {step}: stop_reason={response.stop_reason}")

            if response.stop_reason == "end_turn":
                # Claude finished without calling submit_results — return empty
                print("[JobSearchAgent] Model ended turn without submitting results.")
                return []

            if response.stop_reason != "tool_use":
                print("[JobSearchAgent] Unexpected stop reason; returning no results.")
                return []

            # Process each tool call
            tool_results: list[dict[str, Any]] = []
            final_listings: Optional[list[JobListing]] = None

            for block in response.content:
                if block.type != "tool_use":
                    continue

                if block.name == "submit_results":
                    submitted = len(block.input.get("listings", []))
                    print(
                        f"[JobSearchAgent] Step {step}: submit_results received "
                        f"with {submitted} raw listings."
                    )
                    # Done — parse and return
                    final_listings = self._parse_listings(
                        block.input.get("listings", []), input.max_results
                    )
                    print(
                        f"[JobSearchAgent] Step {step}: parsed "
                        f"{len(final_listings)} deduplicated listings."
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps({"status": "accepted", "count": len(final_listings)}),
                        }
                    )

                elif block.name == "web_search":
                    query = block.input["query"]
                    print(f"[JobSearchAgent] Step {step}: web_search -> {query}")
                    requested_max = block.input.get("max_results", 5)
                    if not isinstance(requested_max, int):
                        requested_max = 5
                    max_results = max(1, min(requested_max, _MAX_SEARCH_RESULTS_PER_TOOL_CALL))
                    result = web_search(
                        query,
                        max_results,
                    )
                    if "error" in result:
                        print(f"[JobSearchAgent] Step {step}: web_search error: {result['error']}")
                    else:
                        print(
                            f"[JobSearchAgent] Step {step}: web_search returned "
                            f"{result.get('count', 0)} results."
                        )
                    compact = self._compact_web_search_result(result)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(compact),
                        }
                    )

                elif block.name == "fetch_url":
                    url = block.input["url"]
                    print(f"[JobSearchAgent] Step {step}: fetch_url -> {url}")
                    result = http_get(url)
                    if "error" in result:
                        print(f"[JobSearchAgent] Step {step}: fetch_url error: {result['error']}")
                    else:
                        print(f"[JobSearchAgent] Step {step}: fetch_url completed.")
                    compact = self._compact_fetch_result(result)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(compact),
                        }
                    )

            # If submit_results was called, we're done
            if final_listings is not None:
                print(f"[JobSearchAgent] Completed with {len(final_listings)} listings.")
                return final_listings

            # Otherwise continue the loop with tool results
            messages.append({"role": "user", "content": tool_results})
            messages = self._trim_messages(messages)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_message_with_retry(self, messages: list[dict[str, Any]], step: int):
        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=_MAX_MODEL_TOKENS,
                    system=[
                        {
                            "type": "text",
                            "text": _SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},  # prompt caching
                        }
                    ],
                    tools=_TOOLS,
                    messages=messages,
                )
            except anthropic.RateLimitError as exc:
                if attempt >= _RATE_LIMIT_RETRIES:
                    raise
                wait_seconds = _RETRY_BASE_SECONDS * (2 ** attempt)
                print(
                    f"[JobSearchAgent] Step {step}: Anthropic rate limit hit; "
                    f"retrying in {wait_seconds}s (attempt {attempt + 1}/{_RATE_LIMIT_RETRIES})."
                )
                # Backoff helps absorb short token-per-minute spikes.
                time.sleep(wait_seconds)

    def _trim_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        max_messages = 1 + (_MAX_CONTEXT_TURNS * 2)
        if len(messages) <= max_messages:
            return messages

        # Keep initial user prompt plus the most recent turns.
        return [messages[0], *messages[-(max_messages - 1):]]

    def _compact_web_search_result(self, result: dict[str, Any]) -> dict[str, Any]:
        if "error" in result:
            return {"error": result.get("error", "unknown"), "results": [], "count": 0}

        compact_results: list[dict[str, str]] = []
        for item in (result.get("results") or [])[:_MAX_SEARCH_RESULTS_FOR_CONTEXT]:
            compact_results.append(
                {
                    "title": (item.get("title") or "")[:200],
                    "url": (item.get("url") or "")[:500],
                    "snippet": (item.get("snippet") or "")[:_MAX_SEARCH_SNIPPET_CHARS],
                }
            )
        return {
            "results": compact_results,
            "count": len(compact_results),
            "truncated": len(result.get("results") or []) > len(compact_results),
        }

    def _compact_fetch_result(self, result: dict[str, Any]) -> dict[str, Any]:
        if "error" in result:
            return {"error": result.get("error", "unknown"), "url": result.get("url", "")}

        text = result.get("text") or ""
        return {
            "url": result.get("url", ""),
            "status_code": result.get("status_code"),
            "content_type": result.get("content_type", ""),
            "text": text[:_MAX_FETCH_TEXT_CHARS_FOR_CONTEXT],
            "truncated": len(text) > _MAX_FETCH_TEXT_CHARS_FOR_CONTEXT,
        }

    def _build_prompt(self, input: JobSearchInput) -> str:
        lines = [f"Job title: {input.job_title}"]
        if input.location:
            lines.append(f"Location: {input.location}")
        else:
            lines.append("Location: any / not specified")
        lines.append(f"Posted within the last {input.date_posted_within_days} days")
        lines.append(f"Collect up to {input.max_results} listings")
        lines.append("\nSearch now and call submit_results when done.")
        return "\n".join(lines)

    def _parse_listings(
        self, raw: list[dict[str, Any]], max_results: int
    ) -> list[JobListing]:
        listings: list[JobListing] = []
        seen_urls: set[str] = set()
        seen_title_company: set[str] = set()
        today = datetime.now().date().isoformat()

        for item in raw:
            url = (item.get("url") or "").strip()
            title = (item.get("title") or "").strip()
            company = (item.get("company") or "").strip()

            if not url or not title:
                continue

            # Deduplicate
            if url in seen_urls:
                continue
            tc_key = f"{title.lower()}|||{company.lower()}"
            if tc_key in seen_title_company:
                continue

            seen_urls.add(url)
            seen_title_company.add(tc_key)

            listings.append(
                JobListing(
                    id=hashlib.md5(url.encode()).hexdigest()[:10],
                    source=_detect_source(url),
                    url=url,
                    title=title,
                    company=company,
                    location=(item.get("location") or "").strip(),
                    posted_date=(item.get("posted_date") or today),
                )
            )

            if len(listings) >= max_results:
                break

        return listings


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------

def _detect_source(url: str):
    u = url.lower()
    if "linkedin.com" in u:
        return "linkedin"
    if "indeed.com" in u:
        return "indeed"
    if "glassdoor.com" in u:
        return "glassdoor"
    if "lever.co" in u:
        return "lever"
    if "greenhouse.io" in u:
        return "greenhouse"
    if "workday.com" in u or "myworkdayjobs.com" in u:
        return "workday"
    return "other"
