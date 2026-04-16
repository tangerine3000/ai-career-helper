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


def main():
    parser = argparse.ArgumentParser(description="CareerMatch — Job Search Agent")
    parser.add_argument("--title",    default="Software Engineer", help="Job title to search for")
    parser.add_argument("--location", default=None,                help="Location (e.g. 'Remote', 'New York')")
    parser.add_argument("--days",     type=int, default=30,        help="Posted within N days")
    parser.add_argument("--max",      type=int, default=20,        help="Max results")
    args = parser.parse_args()

    search_input = JobSearchInput(
        job_title=args.title,
        location=args.location,
        date_posted_within_days=args.days,
        max_results=args.max,
    )

    print(f"\nSearching for: {search_input.job_title}")
    if search_input.location:
        print(f"Location:      {search_input.location}")
    print(f"Max results:   {search_input.max_results}")
    print("-" * 60)

    agent = JobSearchAgent()
    listings = agent.search(search_input)

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
