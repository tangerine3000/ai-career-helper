"""
Test runner for Agent 5 — Advisor Agent.
Also serves as the full end-to-end pipeline runner for all 5 agents.

Mode 1 — load from saved JSON (fastest, skip re-running earlier agents):
    python run_advisor.py \\
        --jd output_job_descriptions.json \\
        --profile output_profile.json \\
        --fit output_fit_report.json

Mode 2 — full pipeline (Agents 1 → 5):
    python run_advisor.py \\
        --title "Senior Backend Engineer" \\
        --location "Remote" \\
        --resume ./my_resume.pdf \\
        --github myusername
"""

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from agents.advisor_agent import AdvisorAgent
from models.types import AdvisorOutput
from utils.loaders import load_fit_report, load_job_descriptions, load_profile


# ---------------------------------------------------------------------------
# Output writers — produce four Markdown files
# ---------------------------------------------------------------------------

def write_outputs(output: AdvisorOutput, out_dir: str = "output") -> list[str]:
    Path(out_dir).mkdir(exist_ok=True)
    written: list[str] = []

    # 1. Improved resume
    resume_path = os.path.join(out_dir, "resume_improved.md")
    with open(resume_path, "w", encoding="utf-8") as f:
        f.write(output.resume_improved)
    written.append(resume_path)

    # 2. Fit report + gap analysis
    fit_path = os.path.join(out_dir, "fit_report.md")
    with open(fit_path, "w", encoding="utf-8") as f:
        f.write(_render_fit_report(output))
    written.append(fit_path)

    # 3. Action plan
    plan_path = os.path.join(out_dir, "action_plan.md")
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(_render_action_plan(output))
    written.append(plan_path)

    # 4. Job links
    links_path = os.path.join(out_dir, "job_links.txt")
    with open(links_path, "w", encoding="utf-8") as f:
        for rec in output.top_job_recommendations:
            f.write(f"[{rec.fit_score:.0f}] {rec.title} @ {rec.company}\n")
            f.write(f"  {rec.url}\n")
            f.write(f"  {rec.why_apply}\n\n")
    written.append(links_path)

    # 5. Full JSON sidecar
    json_path = os.path.join(out_dir, "advisor_output.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_output_to_dict(output), f, indent=2)
    written.append(json_path)

    return written


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------

def _render_fit_report(output: AdvisorOutput) -> str:
    lines: list[str] = ["# Fit Report & Gap Analysis\n"]

    lines.append("## Top Job Recommendations\n")
    for rec in output.top_job_recommendations:
        lines.append(f"### [{rec.fit_score:.0f}/100] {rec.title} @ {rec.company}")
        lines.append(f"{rec.url}")
        lines.append(f"\n{rec.why_apply}\n")

    lines.append("## Critical Skill Gaps\n")
    for gap in output.gap_analysis.critical_gaps:
        lines.append(f"### {gap.skill}")
        lines.append(f"**Why critical:** {gap.why_critical}")
        lines.append(f"\n**How to acquire:** {gap.how_to_acquire}")
        lines.append(f"\n**Time estimate:** {gap.estimated_time}")
        if gap.resources:
            lines.append("\n**Resources:**")
            for r in gap.resources:
                lines.append(f"- {r}")
        lines.append("")

    if output.gap_analysis.nice_to_have_gaps:
        lines.append("## Nice-to-Have Gaps\n")
        for gap in output.gap_analysis.nice_to_have_gaps:
            lines.append(f"- **{gap.skill}:** {gap.suggestion}")
        lines.append("")

    lines.append("## ATS Optimisation Applied\n")
    if output.ats_optimization.keywords_added:
        lines.append(f"**Keywords added:** {', '.join(output.ats_optimization.keywords_added)}\n")
    if output.ats_optimization.formatting_changes:
        lines.append("**Formatting changes:**")
        for change in output.ats_optimization.formatting_changes:
            lines.append(f"- {change}")
        lines.append("")
    if output.ats_optimization.flags:
        lines.append("**Please verify before using:**")
        for flag in output.ats_optimization.flags:
            lines.append(f"- {flag}")
        lines.append("")

    lines.append("## Interview Preparation\n")
    lines.append("These questions are likely to arise given your current gaps:\n")
    for hint in output.interview_prep_hints:
        lines.append(f"- {hint}")

    return "\n".join(lines)


def _render_action_plan(output: AdvisorOutput) -> str:
    lines: list[str] = ["# Action Plan\n"]
    lines.append(
        "Steps are ordered by ROI — highest impact relative to time investment first.\n"
    )

    for item in output.action_plan:
        lines.append(f"## {item.priority}. {item.action}")
        lines.append(f"**Why:** {item.rationale}")
        lines.append(f"\n**Timeline:** {item.timeline}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CareerMatch — Full Pipeline / Advisor Agent")

    # Mode 1: load from files
    parser.add_argument("--jd",      default=None, help="Path to output_job_descriptions.json")
    parser.add_argument("--profile", default=None, help="Path to output_profile.json")
    parser.add_argument("--fit",     default=None, help="Path to output_fit_report.json")

    # Mode 2: full pipeline
    parser.add_argument("--title",    default=None, help="Job title (triggers full pipeline)")
    parser.add_argument("--location", default=None)
    parser.add_argument("--resume",   default=None)
    parser.add_argument("--cv",       default=None)
    parser.add_argument("--github",   default=None)
    parser.add_argument("--linkedin", default=None)
    parser.add_argument("--max",      type=int, default=10)

    # Common
    parser.add_argument("--out", default="output", help="Output directory (default: ./output)")
    args = parser.parse_args()

    # ── Resolve inputs ───────────────────────────────────────────────
    if args.jd and args.profile and args.fit:
        print(f"Loading saved outputs...")
        job_descriptions = load_job_descriptions(args.jd)
        profile = load_profile(args.profile)
        fit_report = load_fit_report(args.fit)
        print(f"  {len(job_descriptions)} job descriptions")
        print(f"  Profile: {profile.name}")
        print(f"  {len(fit_report.per_job)} fit analyses")

    elif args.title and (args.resume or args.github or args.linkedin):
        print("\nRunning full pipeline (Agents 1 → 4)...")

        from agents.job_search_agent import JobSearchAgent
        from agents.job_scraper_agent import JobScraperAgent
        from agents.profile_ingestion_agent import ProfileIngestionAgent
        from agents.fit_analyzer_agent import FitAnalyzerAgent
        from models.types import JobSearchInput, ProfileIngestionInput

        print("Agent 1 — searching jobs...")
        listings = JobSearchAgent().search(
            JobSearchInput(job_title=args.title, location=args.location, max_results=args.max)
        )
        print(f"  {len(listings)} listings found")

        print("Agent 2 — scraping job descriptions...")
        job_descriptions = JobScraperAgent().scrape(listings)
        ok = sum(1 for jd in job_descriptions if not jd.scrape_failed)
        print(f"  {ok}/{len(job_descriptions)} scraped")

        print("Agent 3 — ingesting profile...")
        profile = ProfileIngestionAgent().ingest(ProfileIngestionInput(
            resume_path=args.resume,
            cv_path=args.cv,
            github_username=args.github,
            linkedin_url=args.linkedin,
        ))
        print(f"  Profile built: {len(profile.skills)} skills, {len(profile.experience)} roles")

        print("Agent 4 — analyzing fit...")
        fit_report = FitAnalyzerAgent().analyze(job_descriptions, profile)
        print(f"  Readiness: {fit_report.aggregate.overall_readiness}")

    else:
        parser.print_help()
        print(
            "\nProvide either:\n"
            "  --jd <file> --profile <file> --fit <file>       (load from saved JSON)\n"
            "  --title <title> --resume <file> [--github ...]  (run full pipeline)\n"
        )
        return

    # ── Agent 5 ──────────────────────────────────────────────────────
    print("\nAgent 5 — building recommendations & rewriting resume...")
    print("-" * 60)

    advisor_output = AdvisorAgent().advise(fit_report, profile, job_descriptions)

    # ── Write output files ───────────────────────────────────────────
    written = write_outputs(advisor_output, out_dir=args.out)

    # ── Print summary ────────────────────────────────────────────────
    print(f"\nDone. Output written to ./{args.out}/\n")

    for path in written:
        print(f"  {path}")

    print(f"\nTop recommendations:")
    for rec in advisor_output.top_job_recommendations:
        print(f"  [{rec.fit_score:.0f}] {rec.title} @ {rec.company}")
        print(f"         {rec.why_apply[:100]}")

    print(f"\nCritical gaps to close ({len(advisor_output.gap_analysis.critical_gaps)}):")
    for gap in advisor_output.gap_analysis.critical_gaps:
        print(f"  {gap.skill:<25} ~{gap.estimated_time}")
        print(f"  How: {gap.how_to_acquire[:80]}")

    if advisor_output.ats_optimization.flags:
        print(f"\nPlease verify before submitting your resume:")
        for flag in advisor_output.ats_optimization.flags:
            print(f"  ! {flag}")

    print(f"\nInterview questions to prepare for:")
    for hint in advisor_output.interview_prep_hints:
        print(f"  • {hint[:120]}")


# ---------------------------------------------------------------------------
# JSON serialiser
# ---------------------------------------------------------------------------

def _output_to_dict(o: AdvisorOutput) -> dict:
    return {
        "resume_improved": o.resume_improved,
        "cv_improved": o.cv_improved,
        "gap_analysis": {
            "critical_gaps": [
                {
                    "skill": g.skill,
                    "why_critical": g.why_critical,
                    "how_to_acquire": g.how_to_acquire,
                    "estimated_time": g.estimated_time,
                    "resources": g.resources,
                }
                for g in o.gap_analysis.critical_gaps
            ],
            "nice_to_have_gaps": [
                {"skill": g.skill, "suggestion": g.suggestion}
                for g in o.gap_analysis.nice_to_have_gaps
            ],
        },
        "action_plan": [
            {
                "priority": a.priority,
                "action": a.action,
                "rationale": a.rationale,
                "timeline": a.timeline,
            }
            for a in o.action_plan
        ],
        "ats_optimization": {
            "keywords_added": o.ats_optimization.keywords_added,
            "formatting_changes": o.ats_optimization.formatting_changes,
            "flags": o.ats_optimization.flags,
        },
        "top_job_recommendations": [
            {
                "listing_id": r.listing_id,
                "company": r.company,
                "title": r.title,
                "url": r.url,
                "fit_score": r.fit_score,
                "why_apply": r.why_apply,
            }
            for r in o.top_job_recommendations
        ],
        "interview_prep_hints": o.interview_prep_hints,
    }


if __name__ == "__main__":
    main()
