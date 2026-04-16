from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path

JobSource = Literal["linkedin", "indeed", "glassdoor", "lever", "greenhouse", "workday", "other"]


@dataclass
class JobSearchInput:
    job_title: str
    location: Optional[str] = None
    date_posted_within_days: int = 30
    max_results: int = 20


@dataclass
class JobListing:
    id: str
    source: JobSource
    url: str
    title: str
    company: str
    location: str
    posted_date: str                    # ISO 8601
    raw_html_or_text: Optional[str] = None


@dataclass
class CompensationRange:
    min: Optional[float]
    max: Optional[float]
    currency: str = "USD"


@dataclass
class JobDescription:
    listing_id: str
    title: str
    company: str
    url: str
    responsibilities: list[str]
    required_skills: list[str]
    preferred_skills: list[str]
    tech_stack: list[str]
    keywords: list[str]
    raw_text: str
    required_experience_years: Optional[int] = None
    required_education: Optional[str] = None
    compensation_range: Optional[CompensationRange] = None
    scrape_failed: bool = False
    scrape_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent 3 — Profile Ingestion types
# ---------------------------------------------------------------------------

@dataclass
class ProfileIngestionInput:
    resume_path: Optional[str] = None       # local path: PDF, DOCX, or TXT
    cv_path: Optional[str] = None           # separate CV file (optional)
    linkedin_url: Optional[str] = None      # public LinkedIn profile URL
    github_username: Optional[str] = None
    portfolio_url: Optional[str] = None
    additional_text: Optional[str] = None   # free-form context the user types in


@dataclass
class Contact:
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None


@dataclass
class Skill:
    name: str
    proficiency: Optional[str] = None      # beginner | intermediate | advanced | expert
    years: Optional[float] = None


@dataclass
class WorkExperience:
    title: str
    company: str
    start_date: str                         # ISO 8601 or "Month YYYY"
    end_date: Optional[str] = None          # None means current role
    description: str = ""
    technologies: list[str] = field(default_factory=list)
    achievements: list[str] = field(default_factory=list)


@dataclass
class Education:
    degree: str
    field: str
    institution: str
    graduation_year: Optional[int] = None


@dataclass
class GitHubProject:
    name: str
    description: str
    languages: list[str]
    stars: int
    topics: list[str]
    url: str = ""


@dataclass
class UserProfile:
    name: str
    contact: Contact
    skills: list[Skill]
    experience: list[WorkExperience]
    education: list[Education]
    certifications: list[str]
    github_projects: list[GitHubProject]
    summary: Optional[str] = None
    publications: list[str] = field(default_factory=list)
    raw_resume_text: str = ""               # preserved for Agent 5 rewriting
    sources_used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent 4 — Fit Analyzer types
# ---------------------------------------------------------------------------

@dataclass
class PerJobFit:
    listing_id: str
    title: str
    company: str
    url: str
    fit_score: float                        # 0–100
    matching_skills: list[str]
    missing_required_skills: list[str]
    missing_preferred_skills: list[str]
    keyword_coverage: float                 # 0–100 percent of JD ATS keywords found in profile
    strengths: list[str]                    # areas where candidate clearly exceeds requirements
    summary: str                            # 2–3 sentence human-readable verdict
    experience_gap: Optional[str] = None   # e.g. "JD requires 7 yrs, candidate has 4"
    education_gap: Optional[str] = None


@dataclass
class SkillFrequency:
    skill: str
    frequency: int                          # how many JDs list this as missing


@dataclass
class AggregateFit:
    top_missing_skills: list[SkillFrequency]   # most common gaps across all JDs
    top_matching_skills: list[str]             # skills that match most JDs
    recommended_job_ids: list[str]             # top 3–5 listings by fit score
    overall_readiness: str                     # "strong" | "moderate" | "needs_work"


@dataclass
class FitReport:
    per_job: list[PerJobFit]
    aggregate: AggregateFit


# ---------------------------------------------------------------------------
# Agent 5 — Advisor types
# ---------------------------------------------------------------------------

@dataclass
class CriticalGap:
    skill: str
    why_critical: str           # how many JDs require it, what role it plays
    how_to_acquire: str         # specific actionable path
    estimated_time: str         # e.g. "2–3 weeks"
    resources: list[str] = field(default_factory=list)   # specific courses/docs/projects


@dataclass
class NiceToHaveGap:
    skill: str
    suggestion: str


@dataclass
class GapAnalysis:
    critical_gaps: list[CriticalGap]
    nice_to_have_gaps: list[NiceToHaveGap]


@dataclass
class ActionPlanItem:
    priority: int               # 1 = highest
    action: str                 # concrete, specific step
    rationale: str              # why this has high ROI
    timeline: str               # e.g. "Week 1–2"


@dataclass
class ATSOptimization:
    keywords_added: list[str]           # keywords injected into the improved resume
    formatting_changes: list[str]       # structural improvements made
    flags: list[str] = field(default_factory=list)   # items needing user verification


@dataclass
class JobRecommendation:
    listing_id: str
    company: str
    title: str
    url: str
    fit_score: float
    why_apply: str              # personalised reason for this candidate


@dataclass
class AdvisorOutput:
    resume_improved: str                        # full rewritten resume in Markdown
    gap_analysis: GapAnalysis
    action_plan: list[ActionPlanItem]
    ats_optimization: ATSOptimization
    top_job_recommendations: list[JobRecommendation]
    interview_prep_hints: list[str]             # questions likely to probe gaps
    cv_improved: Optional[str] = None          # set if CV was provided separately
