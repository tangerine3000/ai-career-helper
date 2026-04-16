"""
Test runner for Agent 3 — Profile Ingestion Agent.

Usage examples:
    python run_profile.py --resume ./my_resume.pdf --github myusername
    python run_profile.py --resume ./resume.docx --linkedin https://linkedin.com/in/jane
    python run_profile.py --resume ./resume.pdf --github myuser --linkedin https://linkedin.com/in/jane
    python run_profile.py --text "I am a senior Python engineer with 8 years experience..."
"""

import argparse
import json
from dotenv import load_dotenv

load_dotenv()

from agents.profile_ingestion_agent import ProfileIngestionAgent
from models.types import ProfileIngestionInput


def main():
    parser = argparse.ArgumentParser(description="CareerMatch — Profile Ingestion Agent")
    parser.add_argument("--resume",   default=None, help="Path to resume file (PDF, DOCX, TXT)")
    parser.add_argument("--cv",       default=None, help="Path to separate CV file")
    parser.add_argument("--linkedin", default=None, help="LinkedIn public profile URL")
    parser.add_argument("--github",   default=None, help="GitHub username")
    parser.add_argument("--portfolio",default=None, help="Portfolio URL")
    parser.add_argument("--text",     default=None, help="Additional free-form text")
    args = parser.parse_args()

    if not any([args.resume, args.cv, args.linkedin, args.github, args.text]):
        parser.print_help()
        print("\nError: provide at least one source (--resume, --linkedin, --github, or --text).")
        return

    ingestion_input = ProfileIngestionInput(
        resume_path=args.resume,
        cv_path=args.cv,
        linkedin_url=args.linkedin,
        github_username=args.github,
        portfolio_url=args.portfolio,
        additional_text=args.text,
    )

    print("\nAgent 3 — ingesting profile sources...")
    sources = [k for k, v in vars(ingestion_input).items() if v]
    print(f"Sources: {', '.join(sources)}")
    print("-" * 60)

    profile = ProfileIngestionAgent().ingest(ingestion_input)

    # ── Print summary ────────────────────────────────────────────────
    print(f"\nName:     {profile.name}")
    if profile.contact.email:
        print(f"Email:    {profile.contact.email}")
    if profile.contact.location:
        print(f"Location: {profile.contact.location}")
    if profile.summary:
        print(f"Summary:  {profile.summary[:200]}...")

    print(f"\nSources used: {', '.join(profile.sources_used) or 'none'}")

    print(f"\nSkills ({len(profile.skills)}):")
    for skill in profile.skills[:15]:
        prof = f" [{skill.proficiency}]" if skill.proficiency else ""
        yrs  = f" ({skill.years}y)" if skill.years else ""
        print(f"  {skill.name}{prof}{yrs}")
    if len(profile.skills) > 15:
        print(f"  ... and {len(profile.skills) - 15} more")

    print(f"\nExperience ({len(profile.experience)} roles):")
    for exp in profile.experience:
        end = exp.end_date or "Present"
        print(f"  {exp.title} @ {exp.company}  ({exp.start_date} – {end})")

    print(f"\nEducation ({len(profile.education)}):")
    for edu in profile.education:
        yr = f" ({edu.graduation_year})" if edu.graduation_year else ""
        print(f"  {edu.degree} in {edu.field} — {edu.institution}{yr}")

    if profile.certifications:
        print(f"\nCertifications: {', '.join(profile.certifications)}")

    if profile.github_projects:
        print(f"\nGitHub projects ({len(profile.github_projects)}):")
        for proj in profile.github_projects[:5]:
            langs = ", ".join(proj.languages[:3]) or "—"
            print(f"  {proj.name}  ★{proj.stars}  [{langs}]")
            if proj.description:
                print(f"    {proj.description[:100]}")

    if profile.warnings:
        print(f"\nWarnings:")
        for w in profile.warnings:
            print(f"  ! {w}")

    # ── Save to JSON ─────────────────────────────────────────────────
    output = {
        "name": profile.name,
        "contact": {
            "email":    profile.contact.email,
            "phone":    profile.contact.phone,
            "location": profile.contact.location,
        },
        "summary": profile.summary,
        "skills": [
            {"name": s.name, "proficiency": s.proficiency, "years": s.years}
            for s in profile.skills
        ],
        "experience": [
            {
                "title":        e.title,
                "company":      e.company,
                "start_date":   e.start_date,
                "end_date":     e.end_date,
                "description":  e.description,
                "technologies": e.technologies,
                "achievements": e.achievements,
            }
            for e in profile.experience
        ],
        "education": [
            {
                "degree":          ed.degree,
                "field":           ed.field,
                "institution":     ed.institution,
                "graduation_year": ed.graduation_year,
            }
            for ed in profile.education
        ],
        "certifications": profile.certifications,
        "github_projects": [
            {
                "name":        p.name,
                "description": p.description,
                "languages":   p.languages,
                "stars":       p.stars,
                "topics":      p.topics,
                "url":         p.url,
            }
            for p in profile.github_projects
        ],
        "sources_used": profile.sources_used,
        "warnings":     profile.warnings,
    }

    with open("output_profile.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to output_profile.json")


if __name__ == "__main__":
    main()
