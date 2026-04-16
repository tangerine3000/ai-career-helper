# CareerMatch AI — Multi-Agent Job Fit Application
## Technical Specification

---

## 1. Overview

CareerMatch AI is a multi-agent pipeline that takes a user's job title of interest, autonomously searches live job postings, ingests the user's professional profile (resume, CV, LinkedIn, GitHub), compares both, identifies gaps, and produces actionable output: a gap analysis, a rewritten resume/CV, and a prioritized action plan to maximize interview and hire probability.

---

## 2. Goals & Non-Goals

### Goals
- Automate the full cycle from "I want this job" to "here is what you need to do"
- Surface real, current job postings — not cached or synthetic data
- Produce diffs against the user's actual materials, not generic advice
- Keep every agent independently testable and swappable

### Non-Goals
- Automated job application submission
- Real-time LinkedIn/GitHub OAuth (MVP uses file upload or public URL)
- Interview simulation or coaching

---

## 3. System Architecture

```
User Input
  └─► [Over-Orchestrator]                    ← session manager, planner, user interaction loop
          │
          ├─ validates & enriches input
          ├─ checks session cache
          ├─ emits progress events → UI
          │
          └─► [Pipeline Orchestrator]         ← execution engine for one run
                  │
                  ├─► Agent 1: Job Search       → JobListing[]
                  │
                  ├─► Agent 2: Job Scraper      → JobDescription[]   ─┐ parallel
                  │                                                     │
                  ├─► Agent 3: Profile Ingestion → UserProfile        ─┘
                  │
                  ├─► Agent 4: Fit Analyzer     → FitReport
                  │
                  └─► Agent 5: Advisor          → AdvisorOutput
                                                        │
                                                        └─► [Over-Orchestrator]
                                                                │
                                                        refine? re-run? → loop
                                                                │
                                                           Final Output → User
```

**Two layers of control:**
- The **Over-Orchestrator** owns the session, talks to the user, decides whether to run/re-run/skip sub-pipelines, and enforces budget guardrails.
- The **Pipeline Orchestrator** owns a single end-to-end execution run — it sequences agents, fans out Agent 2, and handles per-run retries.

Each agent receives a typed input and produces a typed output.

---

## 4. Agent Specifications

---

### Agent 1 — Job Search Agent

**Purpose:** Given a job title (and optional location/remote preference), discover current, relevant job postings across multiple sources.

**Inputs:**
```typescript
{
  job_title: string;          // e.g. "Senior Backend Engineer"
  location?: string;          // e.g. "New York" or "Remote"
  date_posted_within_days?: number; // default 30
  max_results?: number;       // default 20
}
```

**Outputs:**
```typescript
JobListing[] = {
  id: string;
  source: "linkedin" | "indeed" | "glassdoor" | "lever" | "greenhouse" | "workday" | "other";
  url: string;
  title: string;
  company: string;
  location: string;
  posted_date: string;        // ISO 8601
  raw_html_or_text?: string;  // for Agent 2 to scrape
}
```

**Behavior:**
- Query multiple job boards via their public APIs or structured search (LinkedIn Jobs, Indeed, Glassdoor, Lever/Greenhouse ATS feeds)
- Deduplicate by URL and (title + company) pair
- Rank by recency; surface top N listings
- Retry with broadened title synonyms if fewer than 5 results returned

**Tools used:** `web_search`, `http_get`

**Failure modes:**
- Rate-limited by source → rotate sources, exponential backoff
- Zero results → expand query (synonyms, broader location)

---

### Agent 2 — Job Scraper Agent

**Purpose:** Pull the full, structured job description from each URL discovered by Agent 1.

**Inputs:** `JobListing[]` from Agent 1

**Outputs:**
```typescript
JobDescription[] = {
  listing_id: string;
  title: string;
  company: string;
  url: string;
  responsibilities: string[];
  required_skills: string[];       // hard requirements
  preferred_skills: string[];      // nice-to-haves
  required_experience_years?: number;
  required_education?: string;
  compensation_range?: { min: number; max: number; currency: string };
  tech_stack: string[];            // extracted technologies
  keywords: string[];              // ATS-relevant terms
  raw_text: string;                // full JD text for embedding
}
```

**Behavior:**
- Fetch full page HTML for each listing URL
- Extract structured fields using LLM extraction pass over raw text
- Normalize skill names (e.g. "Node.js" and "NodeJS" → "Node.js")
- Flag listings where scraping failed (paywalled, JS-rendered without content)

**Tools used:** `http_get`, `html_to_text`, LLM extraction

**Failure modes:**
- JS-rendered pages → use headless browser fallback or skip + log
- Missing fields → mark as `null`, do not fabricate

---

### Agent 3 — Profile Ingestion Agent

**Purpose:** Parse and unify all user-provided professional materials into a single structured profile.

**Inputs:**
```typescript
{
  resume_file?: File;             // PDF, DOCX, TXT
  cv_file?: File;
  linkedin_url?: string;          // public profile URL
  github_username?: string;
  portfolio_url?: string;
  additional_text?: string;       // any extra context from user
}
```

**Outputs:**
```typescript
UserProfile = {
  name: string;
  contact: { email?: string; phone?: string; location?: string };
  summary?: string;
  skills: {
    name: string;
    proficiency?: "beginner" | "intermediate" | "advanced" | "expert";
    years?: number;
  }[];
  experience: {
    title: string;
    company: string;
    start_date: string;
    end_date?: string;            // null = current
    description: string;
    technologies: string[];
    achievements: string[];
  }[];
  education: {
    degree: string;
    field: string;
    institution: string;
    graduation_year?: number;
  }[];
  certifications: string[];
  github_projects: {
    name: string;
    description: string;
    languages: string[];
    stars: number;
    topics: string[];
  }[];
  publications?: string[];
  raw_resume_text: string;        // original text for rewriting
}
```

**Behavior:**
- Parse resume/CV with structure extraction (sections: Summary, Experience, Education, Skills, Projects)
- Fetch LinkedIn public profile HTML and extract equivalent fields
- Call GitHub API (`/users/{username}/repos`) for public repos; extract language breakdown, README summaries, star counts
- Merge all sources, deduplicate skills, prefer most recent/specific data
- Preserve original resume text for diff-based rewriting in Agent 5

**Tools used:** `file_read`, `pdf_parse`, `http_get`, GitHub API, LinkedIn scrape/API

**Failure modes:**
- Private LinkedIn → skip gracefully, note in output
- Unparseable file format → ask user to paste text directly
- GitHub API rate limit → cache or use unauthenticated with reduced rate

---

### Agent 4 — Fit Analyzer Agent

**Purpose:** Compare the set of job descriptions against the user profile and produce a structured gap analysis.

**Inputs:**
- `JobDescription[]` from Agent 2
- `UserProfile` from Agent 3

**Outputs:**
```typescript
FitReport = {
  per_job: {
    listing_id: string;
    title: string;
    company: string;
    fit_score: number;             // 0–100
    matching_skills: string[];
    missing_required_skills: string[];
    missing_preferred_skills: string[];
    experience_gap?: string;       // e.g. "JD requires 7 years, user has 4"
    education_gap?: string;
    keyword_coverage: number;      // % of JD ATS keywords found in profile
    strengths: string[];           // where user clearly exceeds requirements
    summary: string;               // 2–3 sentence human-readable verdict
  }[];
  aggregate: {
    top_missing_skills: { skill: string; frequency: number }[]; // across all JDs
    top_matching_skills: string[];
    recommended_job_ids: string[]; // top 3–5 best-fit listings
    overall_readiness: "strong" | "moderate" | "needs_work";
  };
}
```

**Behavior:**
- For each job, compute fit score using weighted rubric:
  - Required skills coverage: 40%
  - Years of experience match: 25%
  - Preferred skills coverage: 15%
  - ATS keyword coverage: 10%
  - Education match: 10%
- Use semantic similarity (embeddings) for skill matching — "React" covers "React.js", "frontend development" partially covers "UI engineering"
- Aggregate missing skills across all JDs to find the highest-leverage gaps to address
- Sort listings by fit score descending

**Tools used:** LLM with structured output, embedding model for semantic matching

---

### Agent 5 — Advisor Agent

**Purpose:** Turn the fit report into concrete, prioritized, human-ready output: revised resume/CV, a skill-building plan, and interview preparation guidance.

**Inputs:**
- `FitReport` from Agent 4
- `UserProfile` from Agent 3
- `JobDescription[]` from Agent 2 (for keyword injection)

**Outputs:**
```typescript
AdvisorOutput = {
  resume_improved: string;         // full rewritten resume text (Markdown or original format)
  cv_improved?: string;            // if CV provided separately
  gap_analysis: {
    critical_gaps: {               // must-have skills user lacks
      skill: string;
      why_critical: string;
      how_to_acquire: string;      // course, project, certification
      estimated_time: string;      // e.g. "2–4 weeks"
    }[];
    nice_to_have_gaps: {
      skill: string;
      suggestion: string;
    }[];
  };
  action_plan: {
    priority: number;              // 1 = highest
    action: string;                // concrete step
    rationale: string;
    timeline: string;
  }[];
  ats_optimization: {
    keywords_added: string[];      // injected into resume
    formatting_changes: string[];  // e.g. "moved skills section above fold"
  };
  top_job_recommendations: {
    listing_id: string;
    company: string;
    title: string;
    fit_score: number;
    why_apply: string;
  }[];
  interview_prep_hints: string[];  // questions likely to come up given gaps
}
```

**Behavior:**
- Rewrite resume/CV:
  - Inject high-frequency ATS keywords from top job descriptions naturally
  - Reframe existing experience bullets using STAR format where possible
  - Quantify achievements if user provided enough context (ask if not)
  - Do not fabricate experience or skills — only surface/reframe what exists
- For each critical gap:
  - Suggest the fastest credible path (online course, open-source contribution, personal project)
  - Tie suggestion to a specific job requirement
- Action plan ordered by ROI (highest fit-score improvement per unit time)
- Include 3–5 likely interview questions that would probe the user's weakest areas

**Constraints:**
- Never hallucinate credentials, dates, or achievements not present in source material
- Preserve the user's voice in rewrites — do not over-polish into generic HR-speak
- Flag any suggested change with a note if it requires user verification

---

## 5. Over-Orchestrator

**Responsibility:** Top-level controller that owns the user session, manages the full lifecycle across potentially multiple pipeline runs, enforces guardrails, and drives the refinement loop.

### 5.1 Inputs

```typescript
SessionRequest = {
  job_title: string;
  location?: string;
  remote_preference?: "remote" | "hybrid" | "onsite" | "any";
  profile_sources: {
    resume_file?: File;
    cv_file?: File;
    linkedin_url?: string;
    github_username?: string;
    portfolio_url?: string;
    additional_text?: string;
  };
  options?: {
    max_job_listings?: number;     // default 20
    date_posted_within_days?: number; // default 30
    reuse_profile_cache?: boolean; // skip Agent 3 if profile unchanged
    max_llm_cost_usd?: number;     // budget guardrail, default $2.00
  };
}
```

### 5.2 State Machine

```
IDLE
  │  ← SessionRequest received
  ▼
VALIDATING          validate inputs, resolve ambiguous job title (ask user if needed)
  │
  ▼
PLANNING            build execution plan: which agents to run, skip cached stages
  │
  ▼
RUNNING             dispatch Pipeline Orchestrator, stream progress events
  │
  ├─ success ──────►
  │                 REVIEWING        present AdvisorOutput summary to user
  │                     │
  │              user satisfied?
  │               no ─►│◄─ yes
  │                     │              │
  │                 REFINING          FINALIZING → write output files
  │              (loop back to        │
  │               PLANNING with       └─► DONE
  │               updated params)
  │
  └─ fatal error ──► ERROR_RECOVERY → surface to user, suggest fixes
```

### 5.3 Responsibilities

**Input validation & enrichment**
- Normalize job title: strip seniority if too narrow, expand acronyms (e.g. "SRE" → "Site Reliability Engineer")
- If job title is ambiguous (e.g. "engineer"), prompt user to clarify before dispatching
- Validate all file paths and URLs before starting; fail fast with clear messages

**Session caching**
- Cache `UserProfile` per session hash of input files; if files unchanged on re-run, skip Agent 3
- Cache `JobListing[]` for up to 4 hours with the same query parameters; skip Agent 1 if fresh
- Never cache `FitReport` or `AdvisorOutput` — always recompute

**Progress streaming**
- Emit structured progress events to the UI layer throughout execution:

```typescript
ProgressEvent = {
  stage: "validating" | "searching" | "scraping" | "ingesting" | "analyzing" | "advising" | "done" | "error";
  message: string;
  percent_complete: number;    // 0–100
  detail?: string;             // e.g. "scraped 14/20 job listings"
  timestamp: string;
}
```

**Budget guardrails**
- Track estimated LLM token cost as agents run
- Warn user at 80% of `max_llm_cost_usd`; halt at 100% with partial results
- Prefer cheaper models (Haiku) for high-volume extraction tasks (Agent 2 scraping); reserve Sonnet/Opus for Agent 4 and 5

**Concurrency management**
- Agent 2 fan-out: max 5 concurrent scraping tasks (configurable)
- Agents 2 and 3 run in parallel; Agent 4 blocks on both completing
- Global semaphore on LLM calls: max 10 concurrent across all agents

**Refinement loop**
After presenting output, the Over-Orchestrator accepts refinement commands from the user:

| User command | Over-Orchestrator action |
|---|---|
| "Change title to X" | Re-run Agents 1 → 2 → 4 → 5; reuse cached profile |
| "Update my resume" | Re-run Agent 3 → 4 → 5; reuse cached job listings |
| "Show me only remote jobs" | Re-filter existing `JobDescription[]`; re-run Agents 4 → 5 |
| "Focus on company X" | Re-run Agents 4 → 5 with filtered job set |
| "Regenerate resume" | Re-run Agent 5 only |
| "Done" | Finalize and write output files |

**Audit log**
- Write a `session.log` with every agent invocation, input hash, output hash, token count, latency, and any errors
- Enables post-run debugging and cost analysis

### 5.4 Over-Orchestrator Interface

```python
class OverOrchestrator:
    async def start_session(self, request: SessionRequest) -> SessionID
    async def get_progress(self, session_id: SessionID) -> ProgressEvent
    async def get_result(self, session_id: SessionID) -> AdvisorOutput
    async def refine(self, session_id: SessionID, command: RefinementCommand) -> None
    async def cancel(self, session_id: SessionID) -> None
    async def finalize(self, session_id: SessionID, output_dir: Path) -> list[Path]
```

---

## 6. Pipeline Orchestrator

**Responsibility:** Execute one complete agent pipeline run. Sequence agents, pass typed data, handle per-run retries. Spawned and supervised by the Over-Orchestrator.

**Flow:**
```
1. Receive ExecutionPlan from Over-Orchestrator (which stages to run vs. use from cache)
2. Run Agent 1 → validate JobListing[] (skip if cached)
3. Fan out Agent 2 across all listings (max 5 concurrent); run Agent 3 in parallel (skip if cached)
4. Await both; pass results to Agent 4
5. Run Agent 5 with outputs from Agents 2, 3, 4
6. Return PipelineResult to Over-Orchestrator
```

**Error policy:**
- If Agent 1 returns 0 listings → return error to Over-Orchestrator; do not retry internally
- If Agent 2 fails for a listing → skip that listing, log it, continue
- If Agent 3 fails on a source → use partial profile, emit warning event, continue
- If Agent 4 or 5 fail → return error with partial data; Over-Orchestrator decides whether to retry

---

## 7. Data Models (Summary)

| Object | Producer | Consumer |
|---|---|---|
| `SessionRequest` | User / UI | Over-Orchestrator |
| `ExecutionPlan` | Over-Orchestrator | Pipeline Orchestrator |
| `ProgressEvent` | Over-Orchestrator | UI (streamed) |
| `JobListing[]` | Agent 1 | Agent 2, session cache |
| `JobDescription[]` | Agent 2 | Agent 4, Agent 5 |
| `UserProfile` | Agent 3 | Agent 4, Agent 5, session cache |
| `FitReport` | Agent 4 | Agent 5 |
| `AdvisorOutput` | Agent 5 | Over-Orchestrator → User |
| `RefinementCommand` | User / UI | Over-Orchestrator |
| `session.log` | Over-Orchestrator | Developer / audit |

---

## 8. Tech Stack (Recommended)

| Concern | Choice | Rationale |
|---|---|---|
| Agent framework | Claude API with tool use + `claude-sonnet-4-6` | Native multi-tool support, structured output |
| Orchestration | Python `asyncio` or LangGraph | Parallel fan-out for Agent 2, clear state graph |
| PDF parsing | `pdfplumber` or `pymupdf` | Reliable layout extraction |
| Web scraping | `httpx` + `BeautifulSoup` + Playwright fallback | Handles static and JS-rendered pages |
| Embeddings | `text-embedding-3-small` or Cohere | Semantic skill matching |
| Output format | Markdown + JSON sidecar | Human-readable + machine-parseable |
| Storage (session) | In-memory dict or SQLite | MVP; no auth required |

---

## 9. User Interface (MVP)

**CLI first:**
```bash
careermatch --title "Senior Backend Engineer" \
            --resume ./my_resume.pdf \
            --github myusername \
            --linkedin https://linkedin.com/in/myprofile \
            --location "Remote"
```

**Output files written to `./output/`:**
- `fit_report.md` — ranked job listings with per-job gap analysis
- `resume_improved.md` — rewritten resume ready to copy/paste
- `action_plan.md` — prioritized steps with timelines
- `job_links.txt` — URLs of top recommended listings

**Web UI (stretch):** Simple Next.js form with file upload and streaming results display.

---

## 10. Privacy & Security

- No user data persisted beyond the session by default
- Resume/CV files processed in-memory; never uploaded to third-party services
- LinkedIn and GitHub data fetched read-only from public profiles only
- All LLM calls route through the Anthropic API (no data retained per API ToS)
- Option to run fully local with Ollama for sensitive use cases (stretch goal)

---

## 11. Open Questions

| # | Question | Owner | Priority |
|---|---|---|---|
| 1 | Which job boards allow scraping? Need ToS review per source. | Legal/Dev | High |
| 2 | LinkedIn blocks most scraping — use Proxycurl API or user-exported data? | Dev | High |
| 3 | Should resume rewriting show a tracked-changes diff vs. a clean version? | UX | Medium |
| 4 | How do we handle non-English resumes or job postings? | Dev | Low |
| 5 | Rate limit strategy for GitHub API (5000 req/hr authenticated) | Dev | Medium |

---

## 12. Success Metrics

| Metric | Target |
|---|---|
| End-to-end latency (20 job listings) | < 90 seconds |
| Skill extraction accuracy (manual eval sample) | > 85% precision |
| ATS keyword coverage improvement (pre vs. post resume) | +30% minimum |
| User-reported resume quality (5-point scale) | ≥ 4.0 |
| Fit score correlation with actual interview callbacks | Tracked over time |

---

## 13. Milestones

| Milestone | Deliverables |
|---|---|
| M1 — Skeleton | Typed data models, stub agents returning mock data, Over-Orchestrator state machine scaffold |
| M2 — Pipeline Orchestrator | Agent sequencing, parallel fan-out, per-run error policy, `session.log` |
| M3 — Over-Orchestrator core | Session lifecycle, input validation, progress streaming, budget guardrails |
| M4 — Agents 1 & 2 | Live job search + scraping for ≥ 3 sources |
| M5 — Agent 3 | Resume PDF + GitHub ingestion; profile caching in Over-Orchestrator |
| M6 — Agent 4 | Fit scoring with semantic matching |
| M7 — Agent 5 | Resume rewrite + action plan generation |
| M8 — Refinement loop | Over-Orchestrator handles re-run commands; selective stage skipping |
| M9 — CLI MVP | Full pipeline runnable end-to-end with output files and progress display |
| M10 — Polish | LinkedIn integration, web UI, cost reporting (stretch) |
