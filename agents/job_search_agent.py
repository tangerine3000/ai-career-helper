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
                    "description": "Max results to return (1–20). Default 10.",
                    "default": 10,
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
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": self._build_prompt(input)}
        ]

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
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

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Claude finished without calling submit_results — return empty
                return []

            if response.stop_reason != "tool_use":
                return []

            # Process each tool call
            tool_results: list[dict[str, Any]] = []
            final_listings: Optional[list[JobListing]] = None

            for block in response.content:
                if block.type != "tool_use":
                    continue

                if block.name == "submit_results":
                    # Done — parse and return
                    final_listings = self._parse_listings(
                        block.input.get("listings", []), input.max_results
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps({"status": "accepted", "count": len(final_listings)}),
                        }
                    )

                elif block.name == "web_search":
                    result = web_search(
                        block.input["query"],
                        block.input.get("max_results", 10),
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )

                elif block.name == "fetch_url":
                    result = http_get(block.input["url"])
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )

            # If submit_results was called, we're done
            if final_listings is not None:
                return final_listings

            # Otherwise continue the loop with tool results
            messages.append({"role": "user", "content": tool_results})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
