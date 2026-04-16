"""
Agent 3 — Profile Ingestion Agent

Accepts local files (PDF/DOCX/TXT resume or CV), a LinkedIn public profile URL,
a GitHub username, a portfolio URL, and any free-form text the user provides.

All text sources are collected in parallel, combined into a single prompt, and
sent to Claude (Haiku) which extracts a structured UserProfile via tool use.
GitHub repo data is fetched via the API and merged in programmatically — no
LLM pass needed for already-structured data.

LinkedIn note: LinkedIn aggressively blocks scraping. If the fetch fails or
returns a sign-in wall the agent records a warning and continues with the
remaining sources. Users can work around this by:
  1. Exporting their LinkedIn data (Settings → Data Privacy → Get a copy)
     and passing the exported text as `additional_text`.
  2. Using the Proxycurl API (set PROXYCURL_API_KEY env var).
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import anthropic

from models.types import (
    Contact, Education, GitHubProject, ProfileIngestionInput,
    Skill, UserProfile, WorkExperience,
)
from tools.file_parser import parse_file
from tools.github_api import fetch_github_repos
from tools.html_to_text import html_to_text, is_js_rendered
from tools.http_get import http_get

_MAX_SOURCE_CHARS = 10_000   # per source, before sending to Claude
_LINKEDIN_AUTH_HINTS = [
    "authwall", "sign in", "log in", "join linkedin",
    "linkedin.com/login", "linkedin.com/signup",
]

# ---------------------------------------------------------------------------
# Extraction tool
# ---------------------------------------------------------------------------

_EXTRACT_TOOL: dict[str, Any] = {
    "name": "submit_profile",
    "description": (
        "Submit the fully extracted user profile. "
        "Call this exactly once after processing all provided sources."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Full name of the candidate.",
            },
            "contact": {
                "type": "object",
                "properties": {
                    "email":    {"type": ["string", "null"]},
                    "phone":    {"type": ["string", "null"]},
                    "location": {"type": ["string", "null"]},
                },
            },
            "summary": {
                "type": ["string", "null"],
                "description": "Professional summary or objective statement if present.",
            },
            "skills": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":        {"type": "string"},
                        "proficiency": {
                            "type": ["string", "null"],
                            "enum": ["beginner", "intermediate", "advanced", "expert", None],
                        },
                        "years": {"type": ["number", "null"]},
                    },
                    "required": ["name"],
                },
            },
            "experience": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title":        {"type": "string"},
                        "company":      {"type": "string"},
                        "start_date":   {"type": "string"},
                        "end_date":     {"type": ["string", "null"]},
                        "description":  {"type": "string"},
                        "technologies": {"type": "array", "items": {"type": "string"}},
                        "achievements": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["title", "company", "start_date"],
                },
            },
            "education": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "degree":          {"type": "string"},
                        "field":           {"type": "string"},
                        "institution":     {"type": "string"},
                        "graduation_year": {"type": ["integer", "null"]},
                    },
                    "required": ["degree", "field", "institution"],
                },
            },
            "certifications": {
                "type": "array",
                "items": {"type": "string"},
            },
            "publications": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["name", "skills", "experience", "education", "certifications"],
    },
}

# ---------------------------------------------------------------------------
# System prompt (cached)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a professional profile extraction agent. You receive \
combined text from one or more of a candidate's professional sources \
(resume, CV, LinkedIn profile, portfolio, additional notes) and extract a \
structured profile from them.

## Source labels
Input sections are labelled === RESUME ===, === CV ===, === LINKEDIN ===, \
=== PORTFOLIO ===, === ADDITIONAL ===. Where sources overlap, prefer the \
most specific and recent information.

## Skill extraction
- List every distinct skill mentioned anywhere in the sources.
- Infer proficiency only when the source explicitly states it (e.g. "expert in Python", \
"5 years of Java") — otherwise leave null.
- Infer years from date ranges in experience entries when the skill appears there.
- Normalise: "NodeJS"→"Node.js", "k8s"→"Kubernetes", "JS"→"JavaScript", \
"Typescript"→"TypeScript", "golang"→"Go", "Postgres"→"PostgreSQL".

## Experience
- Each distinct role at each company is a separate entry.
- Split description into technologies[] (tools/languages used) and \
achievements[] (quantified outcomes, shipped features, impact).
- end_date null means the role is current.

## Rules
- Extract only what is stated — never infer or fabricate.
- Deduplicate skills across sources.
- Use [] for empty list fields, null for absent scalars.

Call submit_profile exactly once."""

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ProfileIngestionAgent:
    """
    Agent 3 — Profile Ingestion Agent.

    Usage:
        profile = ProfileIngestionAgent().ingest(
            ProfileIngestionInput(
                resume_path="./my_resume.pdf",
                linkedin_url="https://linkedin.com/in/janedoe",
                github_username="janedoe",
            )
        )
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = "claude-haiku-4-5-20251001"
        self._last_extraction_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ingest(self, input: ProfileIngestionInput) -> UserProfile:
        warnings: list[str] = []
        sources_used: list[str] = []
        raw_resume_text = ""
        self._last_extraction_error = None

        # ── 1. Collect text sources in parallel ──────────────────────
        text_sections: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures: dict[str, Any] = {}

            if input.resume_path:
                futures["RESUME"] = pool.submit(parse_file, input.resume_path)
            if input.cv_path:
                futures["CV"] = pool.submit(parse_file, input.cv_path)
            if input.linkedin_url:
                futures["LINKEDIN"] = pool.submit(self._fetch_linkedin, input.linkedin_url)
            if input.portfolio_url:
                futures["PORTFOLIO"] = pool.submit(self._fetch_url_text, input.portfolio_url)

            for label, future in futures.items():
                result = future.result()
                if "error" in result:
                    warnings.append(f"{label}: {result['error']}")
                else:
                    text = result["text"][:_MAX_SOURCE_CHARS]
                    text_sections[label] = text
                    sources_used.append(label.lower())
                    if label in ("RESUME", "CV") and not raw_resume_text:
                        raw_resume_text = result["text"]  # keep full text for Agent 5

        if input.additional_text:
            text_sections["ADDITIONAL"] = input.additional_text[:_MAX_SOURCE_CHARS]
            sources_used.append("additional")

        if not text_sections:
            warnings.append("No sources could be parsed. Returning empty profile.")
            return _empty_profile(warnings)

        # ── 2. Extract structured profile with Claude ────────────────
        combined = _build_combined_text(text_sections)
        extracted = self._extract_profile(combined)

        if extracted is None:
            if self._last_extraction_error:
                warnings.append(
                    f"Claude extraction failed: {self._last_extraction_error}"
                )
            else:
                warnings.append(
                    "Claude extraction failed: no tool output received from model."
                )
            warnings.append("Returning empty profile.")
            return _empty_profile(warnings)

        # ── 3. Fetch GitHub repos (structured — no LLM needed) ───────
        github_projects: list[GitHubProject] = []
        if input.github_username:
            gh_result = fetch_github_repos(input.github_username)
            if "error" in gh_result:
                warnings.append(f"GitHub: {gh_result['error']}")
            else:
                for r in gh_result.get("repos", []):
                    github_projects.append(
                        GitHubProject(
                            name=r["name"],
                            description=r["description"],
                            languages=r["languages"],
                            stars=r["stars"],
                            topics=r["topics"],
                            url=r["url"],
                        )
                    )
                sources_used.append("github")

        # ── 4. Build UserProfile ──────────────────────────────────────
        return _build_profile(
            extracted, github_projects, raw_resume_text, sources_used, warnings
        )

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    def _extract_profile(self, combined_text: str) -> Optional[dict[str, Any]]:
        self._last_extraction_error = None
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "Here are the candidate's professional sources:\n\n"
                    f"{combined_text}\n\n"
                    "Extract the profile and call submit_profile."
                ),
            }
        ]

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[_EXTRACT_TOOL],
                tool_choice={"type": "any"},
                messages=messages,
            )
        except Exception as exc:
            self._last_extraction_error = f"{type(exc).__name__}: {exc}"
            return None

        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_profile":
                return block.input

        self._last_extraction_error = "Model response did not include submit_profile tool call."

        return None

    # ------------------------------------------------------------------
    # Source fetchers
    # ------------------------------------------------------------------

    def _fetch_linkedin(self, url: str) -> dict[str, Any]:
        """
        Try to fetch a public LinkedIn profile page.
        Returns an error dict if blocked or behind auth wall.
        """
        # Try Proxycurl first if API key is available
        proxycurl_key = os.environ.get("PROXYCURL_API_KEY")
        if proxycurl_key:
            return self._fetch_via_proxycurl(url, proxycurl_key)

        result = http_get(url)
        if "error" in result:
            return {"error": f"Could not fetch LinkedIn profile: {result['error']}"}

        text = html_to_text(result.get("text", ""))
        lower = text.lower()

        if any(hint in lower for hint in _LINKEDIN_AUTH_HINTS):
            return {
                "error": (
                    "LinkedIn requires sign-in to view this profile. "
                    "Options: (1) export your LinkedIn data and pass it as additional_text, "
                    "or (2) set PROXYCURL_API_KEY for API-based access."
                )
            }

        if is_js_rendered(text):
            return {"error": "LinkedIn profile page appears JS-rendered and could not be extracted."}

        return {"text": text}

    def _fetch_via_proxycurl(self, linkedin_url: str, api_key: str) -> dict[str, Any]:
        """Fetch LinkedIn profile via Proxycurl API."""
        result = http_get(
            f"https://nubela.co/proxycurl/api/v2/linkedin?url={linkedin_url}",
            extra_headers={"Authorization": f"Bearer {api_key}"},
        )
        if "error" in result:
            return {"error": f"Proxycurl: {result['error']}"}

        try:
            data = json.loads(result["text"])
            # Flatten Proxycurl JSON into readable text for Claude
            text = _proxycurl_to_text(data)
            return {"text": text}
        except Exception as exc:
            return {"error": f"Proxycurl parse error: {exc}"}

    def _fetch_url_text(self, url: str) -> dict[str, Any]:
        result = http_get(url)
        if "error" in result:
            return {"error": result["error"]}
        text = html_to_text(result.get("text", ""))
        if not text.strip():
            return {"error": "Portfolio URL returned no readable text."}
        return {"text": text}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_combined_text(sections: dict[str, str]) -> str:
    parts = []
    for label, text in sections.items():
        parts.append(f"=== {label} ===\n{text.strip()}")
    return "\n\n".join(parts)


def _build_profile(
    data: dict[str, Any],
    github_projects: list[GitHubProject],
    raw_resume_text: str,
    sources_used: list[str],
    warnings: list[str],
) -> UserProfile:
    contact_raw = data.get("contact") or {}
    contact = Contact(
        email=contact_raw.get("email"),
        phone=contact_raw.get("phone"),
        location=contact_raw.get("location"),
    )

    skills = [
        Skill(
            name=s["name"],
            proficiency=s.get("proficiency"),
            years=s.get("years"),
        )
        for s in (data.get("skills") or [])
        if s.get("name")
    ]

    experience = [
        WorkExperience(
            title=e["title"],
            company=e["company"],
            start_date=e["start_date"],
            end_date=e.get("end_date"),
            description=e.get("description", ""),
            technologies=e.get("technologies") or [],
            achievements=e.get("achievements") or [],
        )
        for e in (data.get("experience") or [])
        if e.get("title") and e.get("company")
    ]

    education = [
        Education(
            degree=ed["degree"],
            field=ed["field"],
            institution=ed["institution"],
            graduation_year=ed.get("graduation_year"),
        )
        for ed in (data.get("education") or [])
        if ed.get("degree") and ed.get("institution")
    ]

    return UserProfile(
        name=data.get("name", "Unknown"),
        contact=contact,
        summary=data.get("summary"),
        skills=skills,
        experience=experience,
        education=education,
        certifications=data.get("certifications") or [],
        publications=data.get("publications") or [],
        github_projects=github_projects,
        raw_resume_text=raw_resume_text,
        sources_used=sources_used,
        warnings=warnings,
    )


def _empty_profile(warnings: list[str]) -> UserProfile:
    return UserProfile(
        name="Unknown",
        contact=Contact(),
        skills=[],
        experience=[],
        education=[],
        certifications=[],
        github_projects=[],
        warnings=warnings,
    )


def _proxycurl_to_text(data: dict[str, Any]) -> str:
    """Convert a Proxycurl LinkedIn JSON response to readable text for Claude."""
    lines = []

    if data.get("full_name"):
        lines.append(f"Name: {data['full_name']}")
    if data.get("headline"):
        lines.append(f"Headline: {data['headline']}")
    if data.get("summary"):
        lines.append(f"\nSummary:\n{data['summary']}")
    if data.get("city") or data.get("country_full_name"):
        lines.append(f"Location: {data.get('city', '')} {data.get('country_full_name', '')}".strip())

    experiences = data.get("experiences") or []
    if experiences:
        lines.append("\nExperience:")
        for exp in experiences:
            lines.append(
                f"  {exp.get('title', '')} at {exp.get('company', '')} "
                f"({exp.get('starts_at', {}).get('year', '')}–"
                f"{exp.get('ends_at', {}).get('year', '') or 'Present'})"
            )
            if exp.get("description"):
                lines.append(f"    {exp['description'][:500]}")

    educations = data.get("education") or []
    if educations:
        lines.append("\nEducation:")
        for edu in educations:
            lines.append(
                f"  {edu.get('degree_name', '')} in {edu.get('field_of_study', '')} "
                f"at {edu.get('school', '')} ({edu.get('ends_at', {}).get('year', '')})"
            )

    skills = data.get("skills") or []
    if skills:
        lines.append(f"\nSkills: {', '.join(skills)}")

    certs = data.get("certifications") or []
    if certs:
        lines.append("\nCertifications:")
        for c in certs:
            lines.append(f"  {c.get('name', '')} — {c.get('authority', '')}")

    return "\n".join(lines)
