"""
Test runner for Agent 4 — Fit Analyzer.

Can be driven two ways:

  1. From saved JSON (fast — skips re-running Agents 1–3):
       python run_fit_analyzer.py --jd output_job_descriptions.json --profile output_profile.json

  2. Full pipeline (Agents 1 → 2 → 3 → 4):
       python run_fit_analyzer.py --title "Senior Backend Engineer" --resume ./resume.pdf --github myuser
"""

import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from agents.fit_analyzer_agent import FitAnalyzerAgent
from models.types import (
    CompensationRange, Contact, Education, GitHubProject,
    JobDescription, ProfileIngestionInput, Skill, UserProfile, WorkExperience,
)


# ---------------------------------------------------------------------------
# JSON loaders (rebuild dataclasses from saved output files)
# ---------------------------------------------------------------------------

def load_job_descriptions(path: str) -> list[JobDescription]:
    with open(path) as f:
        raw = json.load(f)
    jds = []
    for item in raw:
        comp = None
        if item.get("compensation_range"):
            c = item["compensation_range"]
            comp = CompensationRange(min=c.get("min"), max=c.get("max"), currency=c.get("currency", "USD"))
        jds.append(JobDescription(
            listing_id=item["listing_id"],
            title=item["title"],
            company=item["company"],
            url=item["url"],
            responsibilities=item.get("responsibilities") or [],
            required_skills=item.get("required_skills") or [],
            preferred_skills=item.get("preferred_skills") or [],
            tech_stack=item.get("tech_stack") or [],
            keywords=item.get("keywords") or [],
            raw_text=item.get("raw_text") or item.get("raw_text", ""),
            required_experience_years=item.get("required_experience_years"),
            required_education=item.get("required_education"),
            compensation_range=comp,
            scrape_failed=item.get("scrape_failed", False),
            scrape_error=item.get("scrape_error"),
        ))
    return jds


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
            Skill(name=s["name"], proficiency=s.get("proficiency"), years=s.get("years"))
            for s in raw.get("skills") or []
        ],
        experience=[
            WorkExperience(
                title=e["title"], company=e["company"], start_date=e["start_date"],
                end_date=e.get("end_date"), description=e.get("description", ""),
                technologies=e.get("technologies") or [],
                achievements=e.get("achievements") or [],
            )
            for e in raw.get("experience") or []
        ],
        education=[
            Education(
                degree=ed["degree"], field=ed["field"], institution=ed["institution"],
                graduation_year=ed.get("graduation_year"),
            )
            for ed in raw.get("education") or []
        ],
        certifications=raw.get("certifications") or [],
        github_projects=[
            GitHubProject(
                name=p["name"], description=p.get("description", ""),
                languages=p.get("languages") or [], stars=p.get("stars", 0),
                topics=p.get("topics") or [], url=p.get("url", ""),
            )
            for p in raw.get("github_projects") or []
        ],
        sources_used=raw.get("sources_used") or [],
        warnings=raw.get("warnings") or [],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CareerMatch — Fit Analyzer Agent")

    # Mode 1: load from files
    parser.add_argument("--jd",      default=None, help="Path to output_job_descriptions.json")
    parser.add_argument("--profile", default=None, help="Path to output_profile.json")

    # Mode 2: full pipeline
    parser.add_argument("--title",    default=None, help="Job title (triggers Agents 1–3 first)")
    parser.add_argument("--location", default=None)
    parser.add_argument("--resume",   default=None)
    parser.add_argument("--github",   default=None)
    parser.add_argument("--linkedin", default=None)
    parser.add_argument("--max",      type=int, default=10)
    args = parser.parse_args()

    # ── Resolve inputs ───────────────────────────────────────────────
    if args.jd and args.profile:
        print(f"Loading job descriptions from {args.jd}")
        print(f"Loading profile from {args.profile}")
        job_descriptions = load_job_descriptions(args.jd)
        profile = load_profile(args.profile)

    elif args.title and (args.resume or args.github or args.linkedin):
        print("Running full pipeline (Agents 1 → 2 → 3)...")
        from agents.job_search_agent import JobSearchAgent
        from agents.job_scraper_agent import JobScraperAgent
        from agents.profile_ingestion_agent import ProfileIngestionAgent
        from models.types import JobSearchInput

        listings = JobSearchAgent().search(
            JobSearchInput(job_title=args.title, location=args.location, max_results=args.max)
        )
        print(f"  Agent 1: {len(listings)} listings found")

        job_descriptions = JobScraperAgent().scrape(listings)
        ok = sum(1 for jd in job_descriptions if not jd.scrape_failed)
        print(f"  Agent 2: {ok}/{len(job_descriptions)} scraped")

        profile = ProfileIngestionAgent().ingest(ProfileIngestionInput(
            resume_path=args.resume,
            github_username=args.github,
            linkedin_url=args.linkedin,
        ))
        print(f"  Agent 3: profile built ({len(profile.skills)} skills, {len(profile.experience)} roles)")

    else:
        parser.print_help()
        print(
            "\nProvide either:\n"
            "  --jd <file> --profile <file>           (load from saved JSON)\n"
            "  --title <title> --resume <file>        (run full pipeline)\n"
        )
        return

    # ── Agent 4 ──────────────────────────────────────────────────────
    valid = sum(1 for jd in job_descriptions if not jd.scrape_failed)
    print(f"\nAgent 4 — analyzing {valid} job descriptions against profile...")
    print("-" * 60)

    report = FitAnalyzerAgent().analyze(job_descriptions, profile)

    # ── Print report ─────────────────────────────────────────────────
    agg = report.aggregate
    print(f"\nOverall readiness: {agg.overall_readiness.upper()}")
    print(f"Jobs analyzed: {len(report.per_job)}")

    print(f"\nTop missing skills (across all JDs):")
    for sf in agg.top_missing_skills[:10]:
        bar = "█" * sf.frequency
        print(f"  {sf.skill:<30} {bar} ({sf.frequency})")

    print(f"\nTop matching skills:")
    print(f"  {', '.join(agg.top_matching_skills[:10])}")

    print(f"\nTop job matches:")
    for job in report.per_job[:5]:
        score_bar = "█" * int(job.fit_score / 10)
        print(f"\n  [{job.fit_score:5.1f}] {score_bar}")
        print(f"         {job.title} @ {job.company}")
        print(f"         {job.summary}")
        if job.missing_required_skills:
            print(f"         Missing: {', '.join(job.missing_required_skills[:5])}")
        if job.strengths:
            print(f"         Strengths: {', '.join(job.strengths[:3])}")

    # ── Save to JSON ─────────────────────────────────────────────────
    output = {
        "aggregate": {
            "overall_readiness": agg.overall_readiness,
            "recommended_job_ids": agg.recommended_job_ids,
            "top_matching_skills": agg.top_matching_skills,
            "top_missing_skills": [
                {"skill": sf.skill, "frequency": sf.frequency}
                for sf in agg.top_missing_skills
            ],
        },
        "per_job": [
            {
                "listing_id": j.listing_id,
                "title": j.title,
                "company": j.company,
                "url": j.url,
                "fit_score": j.fit_score,
                "matching_skills": j.matching_skills,
                "missing_required_skills": j.missing_required_skills,
                "missing_preferred_skills": j.missing_preferred_skills,
                "experience_gap": j.experience_gap,
                "education_gap": j.education_gap,
                "keyword_coverage": j.keyword_coverage,
                "strengths": j.strengths,
                "summary": j.summary,
            }
            for j in report.per_job
        ],
    }

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "output_fit_report.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
