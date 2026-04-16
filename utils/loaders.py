"""
JSON → dataclass loaders shared by all runner scripts.
Each loader rebuilds the corresponding dataclass from a saved output file.
"""

import json
from models.types import (
    AggregateFit, CompensationRange, Contact, Education,
    FitReport, GitHubProject, JobDescription, PerJobFit,
    Skill, SkillFrequency, UserProfile, WorkExperience,
)


def load_job_descriptions(path: str) -> list[JobDescription]:
    with open(path) as f:
        raw = json.load(f)
    result = []
    for item in raw:
        comp = None
        if item.get("compensation_range"):
            c = item["compensation_range"]
            comp = CompensationRange(
                min=c.get("min"), max=c.get("max"), currency=c.get("currency", "USD")
            )
        result.append(JobDescription(
            listing_id=item["listing_id"],
            title=item["title"],
            company=item["company"],
            url=item["url"],
            responsibilities=item.get("responsibilities") or [],
            required_skills=item.get("required_skills") or [],
            preferred_skills=item.get("preferred_skills") or [],
            tech_stack=item.get("tech_stack") or [],
            keywords=item.get("keywords") or [],
            raw_text=item.get("raw_text", ""),
            required_experience_years=item.get("required_experience_years"),
            required_education=item.get("required_education"),
            compensation_range=comp,
            scrape_failed=item.get("scrape_failed", False),
            scrape_error=item.get("scrape_error"),
        ))
    return result


def load_profile(path: str) -> UserProfile:
    with open(path) as f:
        raw = json.load(f)
    contact_raw = raw.get("contact") or {}
    return UserProfile(
        name=raw.get("name", "Unknown"),
        contact=Contact(
            email=contact_raw.get("email"),
            phone=contact_raw.get("phone"),
            location=contact_raw.get("location"),
        ),
        summary=raw.get("summary"),
        skills=[
            Skill(
                name=s["name"],
                proficiency=s.get("proficiency"),
                years=s.get("years"),
            )
            for s in (raw.get("skills") or [])
        ],
        experience=[
            WorkExperience(
                title=e["title"],
                company=e["company"],
                start_date=e["start_date"],
                end_date=e.get("end_date"),
                description=e.get("description", ""),
                technologies=e.get("technologies") or [],
                achievements=e.get("achievements") or [],
            )
            for e in (raw.get("experience") or [])
        ],
        education=[
            Education(
                degree=ed["degree"],
                field=ed["field"],
                institution=ed["institution"],
                graduation_year=ed.get("graduation_year"),
            )
            for ed in (raw.get("education") or [])
        ],
        certifications=raw.get("certifications") or [],
        github_projects=[
            GitHubProject(
                name=p["name"],
                description=p.get("description", ""),
                languages=p.get("languages") or [],
                stars=p.get("stars", 0),
                topics=p.get("topics") or [],
                url=p.get("url", ""),
            )
            for p in (raw.get("github_projects") or [])
        ],
        raw_resume_text=raw.get("raw_resume_text", ""),
        sources_used=raw.get("sources_used") or [],
        warnings=raw.get("warnings") or [],
    )


def load_fit_report(path: str) -> FitReport:
    with open(path) as f:
        raw = json.load(f)

    per_job = [
        PerJobFit(
            listing_id=j["listing_id"],
            title=j["title"],
            company=j["company"],
            url=j["url"],
            fit_score=float(j["fit_score"]),
            matching_skills=j.get("matching_skills") or [],
            missing_required_skills=j.get("missing_required_skills") or [],
            missing_preferred_skills=j.get("missing_preferred_skills") or [],
            keyword_coverage=float(j.get("keyword_coverage", 0)),
            strengths=j.get("strengths") or [],
            summary=j.get("summary", ""),
            experience_gap=j.get("experience_gap"),
            education_gap=j.get("education_gap"),
        )
        for j in (raw.get("per_job") or [])
    ]

    agg_raw = raw.get("aggregate") or {}
    aggregate = AggregateFit(
        top_missing_skills=[
            SkillFrequency(skill=s["skill"], frequency=s["frequency"])
            for s in (agg_raw.get("top_missing_skills") or [])
        ],
        top_matching_skills=agg_raw.get("top_matching_skills") or [],
        recommended_job_ids=agg_raw.get("recommended_job_ids") or [],
        overall_readiness=agg_raw.get("overall_readiness", "needs_work"),
    )

    return FitReport(per_job=per_job, aggregate=aggregate)
