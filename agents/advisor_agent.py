"""
Agent 5 — Advisor Agent

Takes FitReport (Agent 4), UserProfile (Agent 3), and JobDescription[]
(Agent 2) and produces a complete AdvisorOutput.

Two focused Claude Sonnet passes are used to keep each prompt tight:

  Pass 1 — Career Analysis
    Input : FitReport aggregate + top-5 per-job summaries + profile summary
    Output: gap_analysis, action_plan, job_recommendations, interview_prep_hints

  Pass 2 — Resume Rewrite
    Input : original resume text + keywords to inject + gaps from Pass 1
    Output: improved resume (Markdown), keywords_added, formatting_changes, flags

Constraints enforced in both prompts:
  - Never fabricate experience, skills, dates, or achievements
  - Preserve the candidate's voice — avoid over-polished HR-speak
  - Flag any change that requires the candidate's verification
"""

import os
from typing import Any, Optional

import anthropic

from models.types import (
    ActionPlanItem, AdvisorOutput, ATSOptimization, CriticalGap,
    FitReport, GapAnalysis, JobDescription, JobRecommendation,
    NiceToHaveGap, PerJobFit, UserProfile,
)

# ---------------------------------------------------------------------------
# Pass 1 — Career Analysis tool
# ---------------------------------------------------------------------------

_ANALYSIS_TOOL: dict[str, Any] = {
    "name": "submit_career_analysis",
    "description": "Submit the complete career gap analysis, action plan, job recommendations, and interview prep.",
    "input_schema": {
        "type": "object",
        "properties": {
            "gap_analysis": {
                "type": "object",
                "properties": {
                    "critical_gaps": {
                        "type": "array",
                        "description": "Must-have skills the candidate lacks that appear in the majority of target JDs.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "skill":           {"type": "string"},
                                "why_critical":    {
                                    "type": "string",
                                    "description": "How many JDs require it and what role it plays (e.g. 'Required in 8/10 target JDs as the primary container orchestration tool').",
                                },
                                "how_to_acquire":  {
                                    "type": "string",
                                    "description": "Specific, actionable path — not generic advice. Name a real course, project, or certification.",
                                },
                                "estimated_time":  {"type": "string"},
                                "resources": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "2–3 specific resource names or URLs (e.g. 'kubernetes.io/docs/tutorials', 'KodeKloud Kubernetes course').",
                                },
                            },
                            "required": ["skill", "why_critical", "how_to_acquire", "estimated_time"],
                        },
                    },
                    "nice_to_have_gaps": {
                        "type": "array",
                        "description": "Preferred/bonus skills the candidate lacks.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "skill":      {"type": "string"},
                                "suggestion": {"type": "string"},
                            },
                            "required": ["skill", "suggestion"],
                        },
                    },
                },
                "required": ["critical_gaps", "nice_to_have_gaps"],
            },
            "action_plan": {
                "type": "array",
                "description": "Prioritised steps ordered by ROI (highest fit-score improvement per unit time first).",
                "items": {
                    "type": "object",
                    "properties": {
                        "priority":  {"type": "integer", "description": "1 = highest priority."},
                        "action":    {"type": "string",  "description": "Concrete, specific step — not 'improve skills'."},
                        "rationale": {"type": "string",  "description": "Why this step has the highest impact relative to effort."},
                        "timeline":  {"type": "string",  "description": "e.g. 'Week 1–2', 'Month 1'."},
                    },
                    "required": ["priority", "action", "rationale", "timeline"],
                },
            },
            "job_recommendations": {
                "type": "array",
                "description": "Top 3–5 jobs the candidate should prioritise applying to, with personalised reasons.",
                "items": {
                    "type": "object",
                    "properties": {
                        "listing_id": {"type": "string"},
                        "why_apply":  {
                            "type": "string",
                            "description": "Specific reason this role fits this candidate's background and goals.",
                        },
                    },
                    "required": ["listing_id", "why_apply"],
                },
            },
            "interview_prep_hints": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "4–6 interview questions likely to arise given the candidate's gaps. "
                    "Each question should be followed by a one-sentence coaching note on how to answer given the gap."
                ),
            },
        },
        "required": ["gap_analysis", "action_plan", "job_recommendations", "interview_prep_hints"],
    },
}

# ---------------------------------------------------------------------------
# Pass 2 — Resume Rewrite tool
# ---------------------------------------------------------------------------

_RESUME_TOOL: dict[str, Any] = {
    "name": "submit_improved_resume",
    "description": "Submit the fully rewritten resume and a record of all changes made.",
    "input_schema": {
        "type": "object",
        "properties": {
            "resume_improved": {
                "type": "string",
                "description": (
                    "Complete rewritten resume in clean Markdown. "
                    "Must include: header (name, contact, links), summary, skills, "
                    "experience (each role with STAR bullets), education, certifications, "
                    "and optionally projects."
                ),
            },
            "keywords_added": {
                "type": "array",
                "items": {"type": "string"},
                "description": "ATS keywords injected into the resume that were not in the original.",
            },
            "formatting_changes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Structural improvements made, e.g. 'Moved Skills section above Experience'.",
            },
            "flags": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Changes that require the candidate to verify accuracy before using, "
                    "e.g. 'Added AWS to skills — please confirm you have hands-on experience'."
                ),
            },
        },
        "required": ["resume_improved", "keywords_added", "formatting_changes"],
    },
}

# ---------------------------------------------------------------------------
# System prompts (both cached)
# ---------------------------------------------------------------------------

_ANALYSIS_SYSTEM = """You are a senior career coach and job market analyst. \
You receive a candidate's fit report and profile, and you produce a structured \
career development plan.

## Gap analysis rules
- critical_gaps: skills missing from the candidate AND required in ≥ 30% of target JDs.
- nice_to_have_gaps: preferred-only skills or skills appearing in < 30% of JDs.
- For each critical gap, name a SPECIFIC resource — not "take a course" but \
"Complete the official Kubernetes Interactive Tutorial at kubernetes.io/docs/tutorials/".
- Estimate time honestly: a weekend project is "1–2 days", not "2 weeks".

## Action plan rules
- Order strictly by ROI: (number of JDs unlocked) ÷ (hours to acquire).
- Each action must be concrete and completable: "Deploy a containerised Flask app \
to a free-tier AWS EC2 instance using Docker" — not "get cloud experience".
- Tie every action to specific JD requirements from the fit report.
- Limit to 8 items maximum.

## Job recommendation rules
- Recommend only listing_ids that appear in the provided fit report.
- why_apply must reference the candidate's specific background, not generic fit.

## Interview prep rules
- Questions must probe the candidate's actual weakest areas from the fit report.
- After each question, add a coaching note in parentheses on how to answer \
despite the gap (e.g. transferable experience to cite, honest framing to use).
- Format: "Question text (Coaching note: ...)"""

_REWRITE_SYSTEM = """You are an expert technical resume writer. You rewrite \
candidate resumes to maximise ATS keyword coverage and interviewer impact, \
while staying strictly truthful.

## Rewriting rules
1. STAR format: reframe every experience bullet as \
Situation/Task → Action → Result where the result is quantifiable.
2. ATS keywords: inject the provided target keywords naturally into the \
summary, skills, and experience sections. Never keyword-stuff.
3. Preserve voice: keep the candidate's language style — do not replace \
plain English with generic HR phrases like "spearheaded" or "leveraged".
4. No fabrication: only use skills, dates, and achievements present in the \
source material. If you need to add something plausible but unverified, \
add it to the flags list.
5. Skills section: list all confirmed skills, grouped by category \
(Languages, Frameworks, Cloud, Tools, etc.).
6. Summary: 3–4 sentences, lead with the strongest experience, \
end with what the candidate is looking for.
7. Format: clean Markdown — # for name, ## for sections, bullet lists for \
experience items. No tables. No emojis."""

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AdvisorAgent:
    """
    Agent 5 — Advisor Agent.

    Usage:
        output = AdvisorAgent().advise(fit_report, profile, job_descriptions)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = "claude-sonnet-4-6"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def advise(
        self,
        fit_report: FitReport,
        profile: UserProfile,
        job_descriptions: list[JobDescription],
    ) -> AdvisorOutput:
        valid_jds = [jd for jd in job_descriptions if not jd.scrape_failed]
        top_jds   = _select_top_jds(fit_report, valid_jds, n=5)

        # ── Pass 1: career analysis ───────────────────────────────────
        print("  Advisor pass 1/2: gap analysis & action plan...")
        analysis = self._run_analysis(fit_report, profile, top_jds)

        # ── Pass 2: resume rewrite ────────────────────────────────────
        print("  Advisor pass 2/2: rewriting resume...")
        resume_result = self._rewrite_resume(profile, top_jds, fit_report, analysis)

        # ── Assemble output ───────────────────────────────────────────
        return _build_output(analysis, resume_result, fit_report)

    # ------------------------------------------------------------------
    # Pass 1 — Career Analysis
    # ------------------------------------------------------------------

    def _run_analysis(
        self,
        fit_report: FitReport,
        profile: UserProfile,
        top_jds: list[JobDescription],
    ) -> dict[str, Any]:
        context = _build_analysis_context(fit_report, profile, top_jds)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=[{"type": "text", "text": _ANALYSIS_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            tools=[_ANALYSIS_TOOL],
            tool_choice={"type": "tool", "name": "submit_career_analysis"},
            messages=[{"role": "user", "content": context}],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_career_analysis":
                return block.input

        return {}

    # ------------------------------------------------------------------
    # Pass 2 — Resume Rewrite
    # ------------------------------------------------------------------

    def _rewrite_resume(
        self,
        profile: UserProfile,
        top_jds: list[JobDescription],
        fit_report: FitReport,
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        context = _build_rewrite_context(profile, top_jds, fit_report, analysis)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=8096,
            system=[{"type": "text", "text": _REWRITE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            tools=[_RESUME_TOOL],
            tool_choice={"type": "tool", "name": "submit_improved_resume"},
            messages=[{"role": "user", "content": context}],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_improved_resume":
                return block.input

        return {}


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def _build_analysis_context(
    fit_report: FitReport,
    profile: UserProfile,
    top_jds: list[JobDescription],
) -> list[dict[str, Any]]:
    agg = fit_report.aggregate

    profile_block = (
        f"## CANDIDATE PROFILE\n\n"
        f"Name: {profile.name}\n"
        f"Location: {profile.contact.location or 'Not specified'}\n\n"
        f"Skills: {', '.join(s.name for s in profile.skills)}\n\n"
        f"Experience:\n" +
        "\n".join(
            f"  {e.title} @ {e.company} ({e.start_date}–{e.end_date or 'Present'})"
            for e in profile.experience
        ) +
        f"\n\nEducation:\n" +
        "\n".join(
            f"  {ed.degree} in {ed.field} — {ed.institution}"
            for ed in profile.education
        )
    )

    fit_block = (
        f"## FIT REPORT SUMMARY\n\n"
        f"Overall readiness: {agg.overall_readiness}\n"
        f"Jobs analysed: {len(fit_report.per_job)}\n\n"
        f"Top missing skills (by frequency):\n" +
        "\n".join(f"  {sf.skill} — missing in {sf.frequency} JDs" for sf in agg.top_missing_skills[:12]) +
        f"\n\nTop matching skills: {', '.join(agg.top_matching_skills[:10])}\n\n"
        f"Per-job summaries (sorted by fit score):\n" +
        "\n".join(
            f"  [{j.fit_score:.0f}] {j.title} @ {j.company} (id:{j.listing_id})\n"
            f"    {j.summary}\n"
            f"    Missing required: {', '.join(j.missing_required_skills[:5]) or 'none'}\n"
            f"    Strengths: {', '.join(j.strengths[:3]) or 'none'}"
            for j in fit_report.per_job[:10]
        )
    )

    jd_block = "## TOP JOB DESCRIPTIONS\n\n" + "\n\n---\n\n".join(
        f"id:{jd.listing_id}  {jd.title} @ {jd.company}\n"
        f"Required: {', '.join(jd.required_skills[:10])}\n"
        f"Preferred: {', '.join(jd.preferred_skills[:8])}\n"
        f"Tech stack: {', '.join(jd.tech_stack[:10])}\n"
        f"Keywords: {', '.join(jd.keywords[:10])}"
        for jd in top_jds
    )

    return [
        {"type": "text", "text": profile_block, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": fit_block + "\n\n" + jd_block},
    ]


def _build_rewrite_context(
    profile: UserProfile,
    top_jds: list[JobDescription],
    fit_report: FitReport,
    analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    # Collect all target keywords not already prominent in the resume
    all_keywords: list[str] = []
    for jd in top_jds:
        all_keywords.extend(jd.keywords)
        all_keywords.extend(jd.required_skills)
    # Deduplicate while preserving order
    seen: set[str] = set()
    target_keywords: list[str] = []
    for kw in all_keywords:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            target_keywords.append(kw)

    critical_skills = [
        g["skill"]
        for g in (analysis.get("gap_analysis") or {}).get("critical_gaps", [])
    ]

    resume_source = profile.raw_resume_text or _profile_to_resume_text(profile)

    original_block = (
        f"## ORIGINAL RESUME\n\n{resume_source}"
    )

    instructions_block = (
        f"## REWRITE INSTRUCTIONS\n\n"
        f"Candidate name: {profile.name}\n"
        f"Email: {profile.contact.email or 'not provided'}\n"
        f"Location: {profile.contact.location or 'not provided'}\n\n"
        f"Target ATS keywords to inject (use naturally, do not stuff):\n"
        f"{', '.join(target_keywords[:40])}\n\n"
        f"Critical skill gaps to address where honest framing is possible:\n"
        f"{', '.join(critical_skills[:10])}\n\n"
        f"GitHub projects available to cite:\n" +
        "\n".join(
            f"  {p.name} [{', '.join(p.languages[:3])}] — {p.description[:100]}"
            for p in profile.github_projects[:5]
        ) +
        f"\n\nRewrite the resume. Keep it to one page if possible (two max). "
        f"Use STAR format for experience bullets. Do not fabricate anything."
    )

    return [
        {"type": "text", "text": original_block, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": instructions_block},
    ]


# ---------------------------------------------------------------------------
# Output assembler
# ---------------------------------------------------------------------------

def _build_output(
    analysis: dict[str, Any],
    resume_result: dict[str, Any],
    fit_report: FitReport,
) -> AdvisorOutput:
    # Gap analysis
    gap_raw = analysis.get("gap_analysis") or {}
    critical_gaps = [
        CriticalGap(
            skill=g["skill"],
            why_critical=g.get("why_critical", ""),
            how_to_acquire=g.get("how_to_acquire", ""),
            estimated_time=g.get("estimated_time", ""),
            resources=g.get("resources") or [],
        )
        for g in (gap_raw.get("critical_gaps") or [])
    ]
    nice_gaps = [
        NiceToHaveGap(skill=g["skill"], suggestion=g.get("suggestion", ""))
        for g in (gap_raw.get("nice_to_have_gaps") or [])
    ]

    # Action plan
    action_plan = [
        ActionPlanItem(
            priority=a.get("priority", i + 1),
            action=a.get("action", ""),
            rationale=a.get("rationale", ""),
            timeline=a.get("timeline", ""),
        )
        for i, a in enumerate(analysis.get("action_plan") or [])
    ]

    # Job recommendations — enrich with data from FitReport
    fit_by_id = {j.listing_id: j for j in fit_report.per_job}
    job_recs: list[JobRecommendation] = []
    for rec in (analysis.get("job_recommendations") or []):
        lid = rec.get("listing_id", "")
        job = fit_by_id.get(lid)
        if job:
            job_recs.append(JobRecommendation(
                listing_id=lid,
                company=job.company,
                title=job.title,
                url=job.url,
                fit_score=job.fit_score,
                why_apply=rec.get("why_apply", ""),
            ))

    # ATS optimisation
    ats = ATSOptimization(
        keywords_added=resume_result.get("keywords_added") or [],
        formatting_changes=resume_result.get("formatting_changes") or [],
        flags=resume_result.get("flags") or [],
    )

    return AdvisorOutput(
        resume_improved=resume_result.get("resume_improved", ""),
        gap_analysis=GapAnalysis(critical_gaps=critical_gaps, nice_to_have_gaps=nice_gaps),
        action_plan=action_plan,
        ats_optimization=ats,
        top_job_recommendations=job_recs,
        interview_prep_hints=analysis.get("interview_prep_hints") or [],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_top_jds(
    fit_report: FitReport, valid_jds: list[JobDescription], n: int
) -> list[JobDescription]:
    """Return the top-n JDs by fit score."""
    recommended_ids = set(fit_report.aggregate.recommended_job_ids)
    jd_by_id = {jd.listing_id: jd for jd in valid_jds}

    # Preferred: JDs in recommended_job_ids first
    top: list[JobDescription] = []
    for lid in fit_report.aggregate.recommended_job_ids:
        if lid in jd_by_id:
            top.append(jd_by_id[lid])

    # Fill remainder from per_job sorted by score
    for pj in fit_report.per_job:
        if len(top) >= n:
            break
        if pj.listing_id not in recommended_ids and pj.listing_id in jd_by_id:
            top.append(jd_by_id[pj.listing_id])

    return top[:n]


def _profile_to_resume_text(profile: UserProfile) -> str:
    """Fallback: build a plain-text resume from UserProfile when raw_resume_text is absent."""
    lines: list[str] = [profile.name]
    c = profile.contact
    contact_parts = [x for x in [c.email, c.phone, c.location] if x]
    if contact_parts:
        lines.append(" | ".join(contact_parts))

    if profile.summary:
        lines += ["", "SUMMARY", profile.summary]

    if profile.skills:
        lines += ["", "SKILLS", ", ".join(s.name for s in profile.skills)]

    if profile.experience:
        lines.append("\nEXPERIENCE")
        for exp in profile.experience:
            end = exp.end_date or "Present"
            lines.append(f"{exp.title} | {exp.company} | {exp.start_date} – {end}")
            if exp.description:
                lines.append(f"  {exp.description}")
            for ach in exp.achievements:
                lines.append(f"  • {ach}")

    if profile.education:
        lines.append("\nEDUCATION")
        for ed in profile.education:
            yr = f" ({ed.graduation_year})" if ed.graduation_year else ""
            lines.append(f"{ed.degree} in {ed.field} — {ed.institution}{yr}")

    if profile.certifications:
        lines.append(f"\nCERTIFICATIONS\n{', '.join(profile.certifications)}")

    return "\n".join(lines)
