"""
Agent 2 — Job Scraper Agent

Takes JobListing[] from Agent 1, fetches each listing's full page, and uses
Claude (Haiku) to extract a structured JobDescription.

Runs up to MAX_CONCURRENT listings in parallel via ThreadPoolExecutor.
JS-rendered pages that return insufficient text are flagged rather than
silently dropped.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import anthropic

from models.types import CompensationRange, JobDescription, JobListing
from tools.html_to_text import html_to_text, is_js_rendered
from tools.http_get import http_get

MAX_CONCURRENT = 2          # fan-out limit reduced to avoid API rate limits
MAX_TEXT_CHARS = 12_000     # truncate before sending to Claude

# ---------------------------------------------------------------------------
# Extraction tool — Claude calls this to return structured data
# ---------------------------------------------------------------------------

_EXTRACT_TOOL: dict[str, Any] = {
    "name": "submit_job_description",
    "description": (
        "Submit the fully extracted job description. Call this exactly once "
        "after you have read and processed the job listing text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "responsibilities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific duties and responsibilities listed in the JD.",
            },
            "required_skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Hard requirements — must-have skills or qualifications.",
            },
            "preferred_skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Nice-to-haves labelled 'preferred', 'bonus', 'plus', etc.",
            },
            "required_experience_years": {
                "type": ["integer", "null"],
                "description": "Minimum years of experience if explicitly stated, else null.",
            },
            "required_education": {
                "type": ["string", "null"],
                "description": "Education requirement if stated, e.g. \"Bachelor's in CS\". Null if absent.",
            },
            "compensation_range": {
                "type": ["object", "null"],
                "description": "Salary range only if explicitly listed — never infer.",
                "properties": {
                    "min":      {"type": ["number", "null"]},
                    "max":      {"type": ["number", "null"]},
                    "currency": {"type": "string", "default": "USD"},
                },
            },
            "tech_stack": {
                "type": "array",
                "items": {"type": "string"},
                "description": "All specific technologies, frameworks, languages, and tools mentioned.",
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "ATS-relevant terms: job-title variants, domain terms, "
                    "methodology names (e.g. 'Agile', 'CI/CD', 'microservices')."
                ),
            },
        },
        "required": [
            "responsibilities",
            "required_skills",
            "preferred_skills",
            "tech_stack",
            "keywords",
        ],
    },
}

# ---------------------------------------------------------------------------
# System prompt (cached)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a job description extraction agent. You receive the \
plain text of a job listing page and extract structured information from it.

## Skill normalisation
Always use the canonical form:
- "NodeJS" / "node" → "Node.js"
- "ReactJS" / "React.js" → "React"
- "Postgres" / "PostgresQL" / "Postgres DB" → "PostgreSQL"
- "k8s" → "Kubernetes"
- "JS" / "Javascript" → "JavaScript"
- "Typescript" → "TypeScript"
- "golang" / "go lang" → "Go"
- "ML" → "Machine Learning"  (only when used as a skill label, not in sentences)

## Rules
- Extract only what is explicitly stated — do not infer or fabricate.
- Use [] for list fields when nothing is found.
- Use null for optional scalar fields when not present.
- Deduplicate: a skill should not appear in both required_skills and preferred_skills.
- tech_stack should include every specific technology named anywhere in the text.
- keywords should include ATS terms a recruiter would search for.

Call submit_job_description exactly once with the extracted data."""

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class JobScraperAgent:
    """
    Agent 2 — Job Scraper Agent.

    Usage:
        from agents.job_search_agent import JobSearchAgent
        from agents.job_scraper_agent import JobScraperAgent

        listings = JobSearchAgent().search(JobSearchInput(...))
        descriptions = JobScraperAgent().scrape(listings)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = "claude-haiku-4-5-20251001"  # Haiku: cheap for high-volume extraction

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scrape(self, listings: list[JobListing]) -> list[JobDescription]:
        """
        Scrape all listings in parallel (up to MAX_CONCURRENT at a time).
        Always returns one JobDescription per listing — failed ones have
        scrape_failed=True and a populated scrape_error.
        """
        print(
            f"[JobScraperAgent] Starting scrape for {len(listings)} listings "
            f"(max_concurrent={MAX_CONCURRENT})."
        )
        results: list[JobDescription] = []

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
            future_to_listing = {
                pool.submit(self._scrape_one, listing): listing
                for listing in listings
            }
            completed = 0
            for future in as_completed(future_to_listing):
                listing = future_to_listing[future]
                result = future.result()
                completed += 1
                status = "ok" if not result.scrape_failed else f"failed ({result.scrape_error})"
                print(
                    f"[JobScraperAgent] Completed {completed}/{len(listings)}: "
                    f"{listing.title} @ {listing.company} -> {status}"
                )
                results.append(result)

        # Restore original order
        order = {l.id: i for i, l in enumerate(listings)}
        results.sort(key=lambda d: order.get(d.listing_id, 999))
        success_count = sum(1 for r in results if not r.scrape_failed)
        print(
            f"[JobScraperAgent] Done. Successful: {success_count}, "
            f"Failed: {len(results) - success_count}"
        )
        return results

    # ------------------------------------------------------------------
    # Per-listing scrape
    # ------------------------------------------------------------------

    def _scrape_one(self, listing: JobListing) -> JobDescription:
        """Fetch + extract one listing. Never raises — errors are captured."""
        print(f"[JobScraperAgent] Fetching: {listing.title} @ {listing.company}")
        # 1. Fetch the page
        fetch_result = http_get(listing.url)

        if "error" in fetch_result:
            print(
                f"[JobScraperAgent] Fetch failed for {listing.title} @ {listing.company}: "
                f"{fetch_result['error']}"
            )
            return _failed(listing, f"Fetch error: {fetch_result['error']}")

        raw_html = fetch_result.get("text", "")
        plain_text = html_to_text(raw_html)

        # 2. Detect JS-rendered / empty pages
        if is_js_rendered(plain_text):
            print(
                f"[JobScraperAgent] JS-rendered/insufficient content: "
                f"{listing.title} @ {listing.company}"
            )
            return _failed(
                listing,
                "Page appears JS-rendered or returned insufficient text. "
                "Consider Playwright fallback.",
            )

        # 3. Extract with Claude
        print(f"[JobScraperAgent] Extracting structured data: {listing.title} @ {listing.company}")
        return self._extract(listing, plain_text)

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    def _extract(self, listing: JobListing, plain_text: str) -> JobDescription:
        """Single Claude call: pass text, receive structured JobDescription."""
        truncated = plain_text[:MAX_TEXT_CHARS]
        print(
            f"[JobScraperAgent] Sending text to model for {listing.title} @ {listing.company} "
            f"(chars={len(truncated)})."
        )

        user_message = (
            f"Job listing URL: {listing.url}\n"
            f"Title (from search): {listing.title}\n"
            f"Company (from search): {listing.company}\n"
            f"Location (from search): {listing.location}\n\n"
            f"--- JOB LISTING TEXT ---\n{truncated}\n--- END ---\n\n"
            "Extract the job description and call submit_job_description."
        )

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[_EXTRACT_TOOL],
                tool_choice={"type": "any"},   # force a tool call — no prose fallback
                messages=messages,
            )
        except Exception as exc:
            print(f"[JobScraperAgent] Claude API error for {listing.title} @ {listing.company}: {exc}")
            return _failed(listing, f"Claude API error: {exc}")

        # Extract the submit_job_description call
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_job_description":
                print(f"[JobScraperAgent] Extraction complete: {listing.title} @ {listing.company}")
                return _build_description(listing, block.input, plain_text)

        print(f"[JobScraperAgent] Extraction failed: no submit_job_description for {listing.title} @ {listing.company}")
        return _failed(listing, "Claude did not call submit_job_description.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_description(
    listing: JobListing,
    data: dict[str, Any],
    raw_text: str,
) -> JobDescription:
    comp = data.get("compensation_range")
    compensation = None
    if comp and isinstance(comp, dict):
        compensation = CompensationRange(
            min=comp.get("min"),
            max=comp.get("max"),
            currency=comp.get("currency", "USD"),
        )

    return JobDescription(
        listing_id=listing.id,
        title=listing.title,
        company=listing.company,
        url=listing.url,
        responsibilities=data.get("responsibilities") or [],
        required_skills=data.get("required_skills") or [],
        preferred_skills=data.get("preferred_skills") or [],
        tech_stack=data.get("tech_stack") or [],
        keywords=data.get("keywords") or [],
        raw_text=raw_text,
        required_experience_years=data.get("required_experience_years"),
        required_education=data.get("required_education"),
        compensation_range=compensation,
    )


def _failed(listing: JobListing, reason: str) -> JobDescription:
    return JobDescription(
        listing_id=listing.id,
        title=listing.title,
        company=listing.company,
        url=listing.url,
        responsibilities=[],
        required_skills=[],
        preferred_skills=[],
        tech_stack=[],
        keywords=[],
        raw_text="",
        scrape_failed=True,
        scrape_error=reason,
    )
