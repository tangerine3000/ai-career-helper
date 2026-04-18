"""
Quick test runner for Agent 1 — Job Search Agent.

Usage:
    python run_job_search.py
    python run_job_search.py --title "Data Scientist" --location "Remote" --max 10
"""

import argparse
import json
from dotenv import load_dotenv

load_dotenv()

from agents.job_search_agent import JobSearchAgent
from models.types import JobSearchInput


def _parse_locations(raw_location: str | None) -> list[str | None]:
    if not raw_location:
        return [None]
    parts = [p.strip() for p in raw_location.split(",") if p.strip()]
    return parts or [None]


def main():
    parser = argparse.ArgumentParser(description="CareerMatch — Job Search Agent")
    parser.add_argument("--title",    default="Software Engineer", help="Job title to search for")
    parser.add_argument("--location", default=None,                help="Location (e.g. 'Remote', 'New York')")
    parser.add_argument("--days",     type=int, default=30,        help="Posted within N days")
    parser.add_argument("--max",      type=int, default=20,        help="Max results")
    args = parser.parse_args()

    locations = _parse_locations(args.location)

    print(f"\nSearching for: {args.title}")
    if any(locations):
        print(f"Locations:     {', '.join(loc for loc in locations if loc)}")
    print(f"Max results:   {args.max}")
    print("-" * 60)

    agent = JobSearchAgent()
    listings = []
    seen_urls: set[str] = set()

    # If multiple locations are provided, split max budget across them.
    per_location = max(1, args.max // len(locations))
    remainder = max(0, args.max - (per_location * len(locations)))

    for i, location in enumerate(locations):
        budget = per_location + (1 if i < remainder else 0)
        if budget <= 0:
            continue

        label = location or "any / not specified"
        print(f"Searching location: {label} (budget: {budget})")

        search_input = JobSearchInput(
            job_title=args.title,
            location=location,
            date_posted_within_days=args.days,
            max_results=budget,
        )
        location_results = agent.search(search_input)

        for job in location_results:
            if job.url in seen_urls:
                continue
            seen_urls.add(job.url)
            listings.append(job)

        if len(listings) >= args.max:
            break

    listings = listings[:args.max]

    if not listings:
        print("No listings found.")
        return

    print(f"\nFound {len(listings)} listings:\n")
    for i, job in enumerate(listings, 1):
        print(f"{i:>2}. [{job.source.upper()}] {job.title}")
        print(f"     {job.company} — {job.location}")
        print(f"     {job.url}")
        print(f"     Posted: {job.posted_date}")
        print()

    # Write to JSON
    output = [
        {
            "id": j.id,
            "source": j.source,
            "url": j.url,
            "title": j.title,
            "company": j.company,
            "location": j.location,
            "posted_date": j.posted_date,
        }
        for j in listings
    ]
    with open("output_job_listings.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to output_job_listings.json")


if __name__ == "__main__":
    main()
