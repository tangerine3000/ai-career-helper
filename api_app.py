from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_TIMEOUT_SECONDS = 1800
SCRIPT_RUN_CONCURRENCY = 1
SCRIPT_RUN_ACQUIRE_TIMEOUT_SECONDS = 5
_SCRIPT_RUN_SEMAPHORE = threading.BoundedSemaphore(value=SCRIPT_RUN_CONCURRENCY)

OPENAPI_TAGS = [
    {
        "name": "App Health",
        "description": "Health and availability endpoints.",
    },
    {
        "name": "Complete Advisor Run",
        "description": "Run the full end-to-end advisor pipeline.",
    },
    {
        "name": "Individual Agent Runs",
        "description": "Run individual agent scripts or mixed-mode script endpoints.",
    },
]

app = FastAPI(
    title="AI Career Helper Script API",
    version="1.0.0",
    description="HTTP wrapper around run_*.py scripts with one endpoint per runner.",
    openapi_tags=OPENAPI_TAGS,
)


class ScriptRunResponse(BaseModel):
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


class ProfileRunRequest(BaseModel):
    resume: Optional[str] = Field(default=None, description="Path to resume file (PDF, DOCX, TXT)")
    cv: Optional[str] = Field(default=None, description="Path to separate CV file")
    linkedin: Optional[str] = Field(default=None, description="Public LinkedIn profile URL")
    github: Optional[str] = Field(default=None, description="GitHub username")
    portfolio: Optional[str] = Field(default=None, description="Portfolio URL")
    text: Optional[str] = Field(default=None, description="Additional free-form context")

    model_config = {
        "json_schema_extra": {
            "example": {
                "resume": "inputs/Resume.pdf",
                "linkedin": "https://www.linkedin.com/in/example",
                "github": "example-user",
                "text": "Open to AI Engineer and MLOps roles in Ireland.",
            }
        }
    }


class JobSearchRunRequest(BaseModel):
    title: str = Field(default="AI Engineer", description="Job title to search for")
    location: Optional[str] = Field(default=None, description="Location(s), comma-separated, e.g. Remote,Dublin,London")
    days: int = Field(default=30, ge=1, le=365, description="Posted within N days")
    max_results: int = Field(default=20, ge=1, le=50, alias="max")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "title": "AI Engineer",
                "location": "Remote,Dublin,London",
                "days": 14,
                "max": 10,
            }
        },
    }


class ScraperRunRequest(BaseModel):
    title: str = Field(default="AI Engineer", description="Job title to search and scrape")
    location: Optional[str] = Field(default=None, description="Location(s), comma-separated, e.g. Remote,Dublin,London")
    days: int = Field(default=30, ge=1, le=365, description="Posted within N days")
    max_results: int = Field(default=10, ge=1, le=50, alias="max")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "title": "AI Engineer",
                "location": "Remote,Dublin,London",
                "days": 30,
                "max": 8,
            }
        },
    }


class FitAnalyzerRunRequest(BaseModel):
    jd: Optional[str] = Field(default=None, description="Path to job descriptions JSON")
    profile: Optional[str] = Field(default=None, description="Path to profile JSON")
    title: Optional[str] = Field(default=None, description="Job title for full pipeline mode")
    location: Optional[str] = Field(default=None, description="Location(s) for full pipeline mode, comma-separated")
    resume: Optional[str] = Field(default=None, description="Resume path for full pipeline mode")
    github: Optional[str] = Field(default=None, description="GitHub username for full pipeline mode")
    linkedin: Optional[str] = Field(default=None, description="LinkedIn URL for full pipeline mode")
    max_results: int = Field(default=10, ge=1, le=50, alias="max")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "jd": "outputs/output_job_descriptions.json",
                "profile": "outputs/output_profile.json",
                "max": 10,
            }
        },
    }


class AdvisorRunRequest(BaseModel):
    jd: Optional[str] = Field(default=None, description="Path to job descriptions JSON")
    profile: Optional[str] = Field(default=None, description="Path to profile JSON")
    fit: Optional[str] = Field(default=None, description="Path to fit report JSON")
    title: Optional[str] = Field(default=None, description="Job title for full pipeline mode")
    location: Optional[str] = Field(default=None, description="Location(s) for full pipeline mode, comma-separated")
    resume: Optional[str] = Field(default=None, description="Resume path for full pipeline mode")
    cv: Optional[str] = Field(default=None, description="Optional separate CV path")
    github: Optional[str] = Field(default=None, description="GitHub username for full pipeline mode")
    linkedin: Optional[str] = Field(default=None, description="LinkedIn URL for full pipeline mode")
    max_results: int = Field(default=10, ge=1, le=50, alias="max")
    out: str = Field(default="outputs", description="Output directory for advisor artifacts")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "jd": "outputs/output_job_descriptions.json",
                "profile": "outputs/output_profile.json",
                "fit": "outputs/output_fit_report.json",
                "out": "outputs",
            }
        },
    }


class CareerAdvisorFullPipelineRequest(BaseModel):
    title: str = Field(default="AI Engineer", description="Job title for full pipeline mode")
    location: Optional[str] = Field(default=None, description="Location(s) for full pipeline mode, comma-separated")
    resume: Optional[str] = Field(default=None, description="Resume path for full pipeline mode")
    cv: Optional[str] = Field(default=None, description="Optional separate CV path")
    github: Optional[str] = Field(default=None, description="GitHub username for full pipeline mode")
    linkedin: Optional[str] = Field(default=None, description="LinkedIn URL for full pipeline mode")
    max_results: int = Field(default=10, ge=1, le=50, alias="max")
    out: str = Field(default="outputs", description="Output directory for advisor artifacts")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "title": "AI Engineer",
                "location": "Remote,Dublin,London",
                "resume": "inputs/Resume.pdf",
                "github": "example-user",
                "linkedin": "https://www.linkedin.com/in/example",
                "max": 10,
                "out": "outputs",
            }
        },
    }


def _run_script(script_name: str, args: list[str]) -> ScriptRunResponse:
    script_path = APP_ROOT / script_name
    if not script_path.exists():
        raise HTTPException(status_code=500, detail=f"Script not found: {script_name}")

    acquired = _SCRIPT_RUN_SEMAPHORE.acquire(timeout=SCRIPT_RUN_ACQUIRE_TIMEOUT_SECONDS)
    if not acquired:
        raise HTTPException(
            status_code=429,
            detail=(
                "Another pipeline run is already in progress. "
                "Retry shortly to avoid provider rate limits."
            ),
        )

    try:
        cmd = [sys.executable, str(script_path), *args]
        completed = subprocess.run(
            cmd,
            cwd=APP_ROOT,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    finally:
        _SCRIPT_RUN_SEMAPHORE.release()

    return ScriptRunResponse(
        command=cmd,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


@app.get("/health", tags=["App Health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/run/career-advisor/full-pipeline",
    response_model=ScriptRunResponse,
    tags=["Complete Advisor Run"],
)
def run_career_advisor_full_pipeline(payload: CareerAdvisorFullPipelineRequest) -> ScriptRunResponse:
    if not (payload.resume or payload.github or payload.linkedin):
        raise HTTPException(
            status_code=400,
            detail=(
                "Full pipeline mode requires at least one of resume, github, or linkedin."
            ),
        )

    args: list[str] = ["--title", payload.title, "--out", payload.out, "--max", str(payload.max_results)]
    if payload.location:
        args += ["--location", payload.location]
    if payload.resume:
        args += ["--resume", payload.resume]
    if payload.cv:
        args += ["--cv", payload.cv]
    if payload.github:
        args += ["--github", payload.github]
    if payload.linkedin:
        args += ["--linkedin", payload.linkedin]

    return _run_script("run_advisor.py", args)


@app.post("/run/career-advisor", response_model=ScriptRunResponse, tags=["Individual Agent Runs"])
def run_career_advisor(payload: AdvisorRunRequest) -> ScriptRunResponse:
    from_saved = bool(payload.jd and payload.profile and payload.fit)
    full_pipeline = bool(payload.title and (payload.resume or payload.github or payload.linkedin))

    if not (from_saved or full_pipeline):
        raise HTTPException(
            status_code=400,
            detail=(
                "Provide either (jd + profile + fit) for saved-input mode, or "
                "(title + one of resume/github/linkedin) for full pipeline mode."
            ),
        )

    args: list[str] = ["--out", payload.out, "--max", str(payload.max_results)]
    if payload.jd:
        args += ["--jd", payload.jd]
    if payload.profile:
        args += ["--profile", payload.profile]
    if payload.fit:
        args += ["--fit", payload.fit]
    if payload.title:
        args += ["--title", payload.title]
    if payload.location:
        args += ["--location", payload.location]
    if payload.resume:
        args += ["--resume", payload.resume]
    if payload.cv:
        args += ["--cv", payload.cv]
    if payload.github:
        args += ["--github", payload.github]
    if payload.linkedin:
        args += ["--linkedin", payload.linkedin]

    return _run_script("run_advisor.py", args)


@app.post("/run/fit-analyzer", response_model=ScriptRunResponse, tags=["Individual Agent Runs"])
def run_fit_analyzer(payload: FitAnalyzerRunRequest) -> ScriptRunResponse:
    from_saved = bool(payload.jd and payload.profile)
    full_pipeline = bool(payload.title and (payload.resume or payload.github or payload.linkedin))

    if not (from_saved or full_pipeline):
        raise HTTPException(
            status_code=400,
            detail=(
                "Provide either (jd + profile) for saved-input mode, or "
                "(title + one of resume/github/linkedin) for full pipeline mode."
            ),
        )

    args: list[str] = ["--max", str(payload.max_results)]
    if payload.jd:
        args += ["--jd", payload.jd]
    if payload.profile:
        args += ["--profile", payload.profile]
    if payload.title:
        args += ["--title", payload.title]
    if payload.location:
        args += ["--location", payload.location]
    if payload.resume:
        args += ["--resume", payload.resume]
    if payload.github:
        args += ["--github", payload.github]
    if payload.linkedin:
        args += ["--linkedin", payload.linkedin]

    return _run_script("run_fit_analyzer.py", args)


@app.post("/run/job-scraper", response_model=ScriptRunResponse, tags=["Individual Agent Runs"])
def run_scraper(payload: ScraperRunRequest) -> ScriptRunResponse:
    args: list[str] = ["--title", payload.title, "--days", str(payload.days), "--max", str(payload.max_results)]
    if payload.location:
        args += ["--location", payload.location]
    return _run_script("run_scraper.py", args)


@app.post("/run/job-search", response_model=ScriptRunResponse, tags=["Individual Agent Runs"])
def run_job_search(payload: JobSearchRunRequest) -> ScriptRunResponse:
    args: list[str] = ["--title", payload.title, "--days", str(payload.days), "--max", str(payload.max_results)]
    if payload.location:
        args += ["--location", payload.location]
    return _run_script("run_job_search.py", args)


@app.post("/run/profile-ingester", response_model=ScriptRunResponse, tags=["Individual Agent Runs"])
def run_profile_ingester(payload: ProfileRunRequest) -> ScriptRunResponse:
    if not any([payload.resume, payload.cv, payload.linkedin, payload.github, payload.text]):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one source: resume, cv, linkedin, github, or text.",
        )

    args: list[str] = []
    if payload.resume:
        args += ["--resume", payload.resume]
    if payload.cv:
        args += ["--cv", payload.cv]
    if payload.linkedin:
        args += ["--linkedin", payload.linkedin]
    if payload.github:
        args += ["--github", payload.github]
    if payload.portfolio:
        args += ["--portfolio", payload.portfolio]
    if payload.text:
        args += ["--text", payload.text]
    return _run_script("run_profile.py", args)
