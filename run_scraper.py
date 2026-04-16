"""
Test runner for Agent 1 (Job Search) → Agent 2 (Job Scraper).

Usage:
    python run_scraper.py
    python run_scraper.py --title "Data Scientist" --location "Remote" --max 5
"""

import argparse
import json
from dotenv import load_dotenv

load_dotenv()

from agents.job_search_agent import JobSearchAgent
from agents.job_scraper_agent import JobScraperAgent
from models.types import JobSearchInput


def main():
    parser = argparse.ArgumentParser(description="CareerMatch — Search + Scrape")
    parser.add_argument("--title",    default="Software Engineer")
    parser.add_argument("--location", default=None)
    parser.add_argument("--days",     type=int, default=30)
    parser.add_argument("--max",      type=int, default=10)
    args = parser.parse_args()

    search_input = JobSearchInput(
        job_title=args.title,
        location=args.location,
        date_posted_within_days=args.days,
        max_results=args.max,
    )

    # --- Agent 1: search ---
    print(f"\nAgent 1 — searching for: {search_input.job_title}")
    print("-" * 60)
    listings = JobSearchAgent().search(search_input)

    if not listings:
        print("No listings found. Exiting.")
        return

    print(f"Found {len(listings)} listings.\n")

    # --- Agent 2: scrape ---
    print(f"Agent 2 — scraping {len(listings)} listings (up to 5 concurrent)...")
    print("-" * 60)
    descriptions = JobScraperAgent().scrape(listings)

    ok  = [d for d in descriptions if not d.scrape_failed]
    bad = [d for d in descriptions if d.scrape_failed]

    print(f"\nScraped: {len(ok)} OK  |  {len(bad)} failed\n")

    for jd in ok:
        print(f"  {jd.title} @ {jd.company}")
        print(f"    Required skills:  {', '.join(jd.required_skills[:6]) or '—'}")
        print(f"    Tech stack:       {', '.join(jd.tech_stack[:6]) or '—'}")
        if jd.required_experience_years:
            print(f"    Experience:       {jd.required_experience_years} yrs")
        if jd.compensation_range:
            c = jd.compensation_range
            print(f"    Compensation:     {c.currency} {c.min}–{c.max}")
        print()

    if bad:
        print("Failed listings:")
        for jd in bad:
            print(f"  {jd.title} @ {jd.company} — {jd.scrape_error}")

    # Write results to JSON
    output = []
    for jd in descriptions:
        entry = {
            "listing_id": jd.listing_id,
            "title": jd.title,
            "company": jd.company,
            "url": jd.url,
            "scrape_failed": jd.scrape_failed,
            "scrape_error": jd.scrape_error,
        }
        if not jd.scrape_failed:
            entry.update({
                "responsibilities": jd.responsibilities,
                "required_skills": jd.required_skills,
                "preferred_skills": jd.preferred_skills,
                "tech_stack": jd.tech_stack,
                "keywords": jd.keywords,
                "required_experience_years": jd.required_experience_years,
                "required_education": jd.required_education,
                "compensation_range": (
                    {
                        "min": jd.compensation_range.min,
                        "max": jd.compensation_range.max,
                        "currency": jd.compensation_range.currency,
                    }
                    if jd.compensation_range else None
                ),
            })
        output.append(entry)

    with open("output_job_descriptions.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to output_job_descriptions.json")


if __name__ == "__main__":
    main()
