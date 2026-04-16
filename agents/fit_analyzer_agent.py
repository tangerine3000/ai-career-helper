"""
Agent 4 — Fit Analyzer Agent

Compares a set of JobDescription objects (from Agent 2) against a UserProfile
(from Agent 3) and produces a FitReport with per-job scores and aggregate stats.

Scoring rubric (encoded in the system prompt and enforced structurally):
  Required skills coverage   40 %
  Years of experience match  25 %
  Preferred skills coverage  15 %
  ATS keyword coverage       10 %
  Education match            10 %

Claude (Sonnet) handles semantic skill matching — it recognises that "ReactJS"
satisfies "React", or that "backend systems" experience partially satisfies
"distributed systems" — so no separate embedding model is needed.

Each JD is analysed in a separate parallel Claude call.  The user profile is
serialised to text and cached as the first user message so it is only billed
once across all parallel calls.
"""

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import anthropic

from models.types import (
    AggregateFit, FitReport, JobDescription, PerJobFit,
    SkillFrequency, UserProfile,
)

MAX_CONCURRENT = 5

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

_ANALYZE_TOOL: dict[str, Any] = {
    "name": "submit_fit_analysis",
    "description": (
        "Submit the completed fit analysis for one job description. "
        "Call this exactly once after you have scored all dimensions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "fit_score": {
                "type": "number",
                "description": (
                    "Overall fit score 0–100 computed from the weighted rubric:\n"
                    "  required_skills_score  × 0.40\n"
                    "  experience_score       × 0.25\n"
                    "  preferred_skills_score × 0.15\n"
                    "  keyword_coverage_score × 0.10\n"
                    "  education_score        × 0.10\n"
                    "Each sub-score is 0–100 before weighting."
                ),
            },
            "matching_skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Skills from the JD (required or preferred) that the candidate "
                    "demonstrably has, including semantic matches."
                ),
            },
            "missing_required_skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Hard-requirement skills the candidate lacks.",
            },
            "missing_preferred_skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Preferred/nice-to-have skills the candidate lacks.",
            },
            "experience_gap": {
                "type": ["string", "null"],
                "description": (
                    "Human-readable gap description if the candidate's years of "
                    "experience fall short, e.g. 'JD requires 7 yrs, candidate has 4'. "
                    "Null if no gap."
                ),
            },
            "education_gap": {
                "type": ["string", "null"],
                "description": (
                    "Human-readable gap if the candidate's education does not meet "
                    "the stated requirement. Null if no gap or no requirement stated."
                ),
            },
            "keyword_coverage": {
                "type": "number",
                "description": (
                    "Percentage (0–100) of the JD's ATS keywords that appear in "
                    "the candidate's profile (skills, job titles, achievements)."
                ),
            },
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "2–5 concrete areas where the candidate clearly meets or exceeds "
                    "what the JD asks for."
                ),
            },
            "summary": {
                "type": "string",
                "description": (
                    "2–3 sentence verdict: overall fit, biggest strength, biggest gap."
                ),
            },
        },
        "required": [
            "fit_score",
            "matching_skills",
            "missing_required_skills",
            "missing_preferred_skills",
            "keyword_coverage",
            "strengths",
            "summary",
        ],
    },
}

# ---------------------------------------------------------------------------
# System prompt (cached — stable for every per-job call)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a job fit analysis expert. You receive a candidate \
profile and a single job description, and you produce a structured fit analysis.

## Scoring rubric
Compute each sub-score (0–100) then apply weights to get the final fit_score:

| Dimension                | Weight | How to score |
|--------------------------|--------|--------------|
| Required skills coverage | 40 %   | (matched required skills) / (total required skills) × 100 |
| Experience years match   | 25 %   | min(candidate_years / required_years, 1.0) × 100; 100 if no requirement |
| Preferred skills coverage| 15 %   | (matched preferred skills) / (total preferred skills) × 100; 100 if none listed |
| ATS keyword coverage     | 10 %   | (JD keywords found in profile) / (total JD keywords) × 100 |
| Education match          | 10 %   | 100 if requirement met or not stated; 50 if related field; 0 if clearly unmet |

## Semantic matching rules
- Match skills semantically, not just literally.
  Examples: "ReactJS" satisfies "React"; "Node.js" satisfies "Node";
  "led a team of engineers" satisfies "people management";
  "built REST APIs" partially satisfies "API design".
- A skill in the candidate's GitHub projects or job technologies counts.
- Experience in a closely related area scores partial credit (50 %) for that skill.

## Experience years
- Sum all relevant work experience years from the candidate's experience list.
- Only count roles where the technologies/domain overlap with the JD.

## Rules
- Base every judgement on the provided profile — never assume skills not listed.
- Use the canonical skill name from the JD in missing_required_skills and
  missing_preferred_skills.
- Strengths should be specific and tied to JD requirements, not generic praise.
- Summary must mention the top strength and the single most important gap."""

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class FitAnalyzerAgent:
    """
    Agent 4 — Fit Analyzer Agent.

    Usage:
        report = FitAnalyzerAgent().analyze(job_descriptions, user_profile)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        # Sonnet: reserved for Agent 4 and 5 (needs stronger reasoning)
        self.model = "claude-sonnet-4-6"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(
        self,
        job_descriptions: list[JobDescription],
        profile: UserProfile,
    ) -> FitReport:
        """
        Score every non-failed JobDescription against the UserProfile.
        Returns a FitReport sorted by fit_score descending.
        """
        # Only analyse successfully scraped JDs
        valid_jds = [jd for jd in job_descriptions if not jd.scrape_failed]

        if not valid_jds:
            return FitReport(
                per_job=[],
                aggregate=AggregateFit(
                    top_missing_skills=[],
                    top_matching_skills=[],
                    recommended_job_ids=[],
                    overall_readiness="needs_work",
                ),
            )

        profile_text = _profile_to_text(profile)

        # Fan out: one Claude call per JD, max MAX_CONCURRENT in parallel
        per_job_results: list[PerJobFit] = []

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
            future_to_jd = {
                pool.submit(self._analyze_one, jd, profile_text): jd
                for jd in valid_jds
            }
            for future in as_completed(future_to_jd):
                result = future.result()
                if result:
                    per_job_results.append(result)

        # Sort by fit_score descending
        per_job_results.sort(key=lambda r: r.fit_score, reverse=True)

        aggregate = _compute_aggregate(per_job_results)
        return FitReport(per_job=per_job_results, aggregate=aggregate)

    # ------------------------------------------------------------------
    # Per-JD analysis
    # ------------------------------------------------------------------

    def _analyze_one(
        self, jd: JobDescription, profile_text: str
    ) -> Optional[PerJobFit]:
        """Send one JD + the cached profile to Claude and return PerJobFit."""
        jd_text = _jd_to_text(jd)

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    # Profile cached as first content block — shared prefix across all calls
                    {
                        "type": "text",
                        "text": f"## CANDIDATE PROFILE\n\n{profile_text}",
                        "cache_control": {"type": "ephemeral"},
                    },
                    # JD is unique per call — not cached
                    {
                        "type": "text",
                        "text": f"## JOB DESCRIPTION TO ANALYZE\n\n{jd_text}\n\nAnalyse this job against the candidate profile above and call submit_fit_analysis.",
                    },
                ],
            }
        ]

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
                tools=[_ANALYZE_TOOL],
                tool_choice={"type": "tool", "name": "submit_fit_analysis"},
                messages=messages,
            )
        except Exception as exc:
            print(f"  [warn] fit analysis failed for {jd.title} @ {jd.company}: {exc}")
            return None

        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_fit_analysis":
                return _build_per_job_fit(jd, block.input)

        return None


# ---------------------------------------------------------------------------
# Aggregate computation (pure Python — no LLM needed)
# ---------------------------------------------------------------------------

def _compute_aggregate(results: list[PerJobFit]) -> AggregateFit:
    if not results:
        return AggregateFit(
            top_missing_skills=[],
            top_matching_skills=[],
            recommended_job_ids=[],
            overall_readiness="needs_work",
        )

    # Missing skill frequency across all JDs
    missing_counter: Counter = Counter()
    for r in results:
        for skill in r.missing_required_skills:
            missing_counter[skill.lower()] += 1

    # Restore canonical capitalisation from the first occurrence
    canonical: dict[str, str] = {}
    for r in results:
        for skill in r.missing_required_skills:
            canonical.setdefault(skill.lower(), skill)

    top_missing = [
        SkillFrequency(skill=canonical[skill], frequency=count)
        for skill, count in missing_counter.most_common(15)
    ]

    # Matching skill frequency
    matching_counter: Counter = Counter()
    for r in results:
        for skill in r.matching_skills:
            matching_counter[skill.lower()] += 1
    top_matching_canonical: dict[str, str] = {}
    for r in results:
        for skill in r.matching_skills:
            top_matching_canonical.setdefault(skill.lower(), skill)
    top_matching = [
        top_matching_canonical[s]
        for s, _ in matching_counter.most_common(10)
    ]

    # Top recommended jobs
    recommended = [r.listing_id for r in results[:5]]

    # Overall readiness from average score
    avg = sum(r.fit_score for r in results) / len(results)
    if avg >= 70:
        readiness = "strong"
    elif avg >= 45:
        readiness = "moderate"
    else:
        readiness = "needs_work"

    return AggregateFit(
        top_missing_skills=top_missing,
        top_matching_skills=top_matching,
        recommended_job_ids=recommended,
        overall_readiness=readiness,
    )


# ---------------------------------------------------------------------------
# Serialisers
# ---------------------------------------------------------------------------

def _profile_to_text(p: UserProfile) -> str:
    lines: list[str] = []

    lines.append(f"Name: {p.name}")
    if p.contact.location:
        lines.append(f"Location: {p.contact.location}")
    if p.summary:
        lines.append(f"\nSummary:\n{p.summary}")

    if p.skills:
        lines.append("\nSkills:")
        for s in p.skills:
            parts = [s.name]
            if s.proficiency:
                parts.append(f"[{s.proficiency}]")
            if s.years:
                parts.append(f"({s.years} yrs)")
            lines.append(f"  - {' '.join(parts)}")

    if p.experience:
        lines.append("\nWork Experience:")
        for exp in p.experience:
            end = exp.end_date or "Present"
            lines.append(f"\n  {exp.title} @ {exp.company}  ({exp.start_date} – {end})")
            if exp.description:
                lines.append(f"  {exp.description}")
            if exp.technologies:
                lines.append(f"  Technologies: {', '.join(exp.technologies)}")
            if exp.achievements:
                for ach in exp.achievements:
                    lines.append(f"    • {ach}")

    if p.education:
        lines.append("\nEducation:")
        for ed in p.education:
            yr = f" ({ed.graduation_year})" if ed.graduation_year else ""
            lines.append(f"  {ed.degree} in {ed.field} — {ed.institution}{yr}")

    if p.certifications:
        lines.append(f"\nCertifications: {', '.join(p.certifications)}")

    if p.github_projects:
        lines.append("\nGitHub Projects:")
        for proj in p.github_projects[:8]:
            langs = ", ".join(proj.languages[:4])
            lines.append(f"  {proj.name} [{langs}] ★{proj.stars}")
            if proj.description:
                lines.append(f"    {proj.description[:150]}")

    return "\n".join(lines)


def _jd_to_text(jd: JobDescription) -> str:
    lines: list[str] = []

    lines.append(f"Title:   {jd.title}")
    lines.append(f"Company: {jd.company}")
    lines.append(f"URL:     {jd.url}")

    if jd.required_experience_years:
        lines.append(f"Required experience: {jd.required_experience_years} years")
    if jd.required_education:
        lines.append(f"Required education: {jd.required_education}")

    if jd.responsibilities:
        lines.append("\nResponsibilities:")
        for r in jd.responsibilities:
            lines.append(f"  - {r}")

    if jd.required_skills:
        lines.append(f"\nRequired skills: {', '.join(jd.required_skills)}")

    if jd.preferred_skills:
        lines.append(f"Preferred skills: {', '.join(jd.preferred_skills)}")

    if jd.tech_stack:
        lines.append(f"Tech stack: {', '.join(jd.tech_stack)}")

    if jd.keywords:
        lines.append(f"ATS keywords: {', '.join(jd.keywords)}")

    if jd.compensation_range:
        c = jd.compensation_range
        lines.append(f"Compensation: {c.currency} {c.min}–{c.max}")

    return "\n".join(lines)


def _build_per_job_fit(jd: JobDescription, data: dict[str, Any]) -> PerJobFit:
    return PerJobFit(
        listing_id=jd.listing_id,
        title=jd.title,
        company=jd.company,
        url=jd.url,
        fit_score=float(data.get("fit_score", 0)),
        matching_skills=data.get("matching_skills") or [],
        missing_required_skills=data.get("missing_required_skills") or [],
        missing_preferred_skills=data.get("missing_preferred_skills") or [],
        experience_gap=data.get("experience_gap"),
        education_gap=data.get("education_gap"),
        keyword_coverage=float(data.get("keyword_coverage", 0)),
        strengths=data.get("strengths") or [],
        summary=data.get("summary", ""),
    )
