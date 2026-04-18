# AI Career Helper

AI Career Helper is a multi-agent Python application that:

- finds relevant jobs,
- scrapes and structures job descriptions,
- ingests candidate profile sources,
- analyzes job fit and skill gaps,
- generates advisor outputs (improved resume, action plan, recommendations).

The project supports:

- CLI runners (`run_*.py`) for each stage,
- a FastAPI wrapper (`api_app.py`) with one endpoint per runner.

## Quick Start

Fastest happy path from a fresh checkout:

```bash
pip install -r requirements.txt
py run_profile.py --resume "inputs/Resume.pdf" --github your-github-username
py run_scraper.py --title "AI Engineer" --location "Remote" --max 10
py run_advisor.py --jd outputs/output_job_descriptions.json --profile outputs/output_profile.json --fit outputs/output_fit_report.json --out outputs
```

Primary outputs are written under `outputs/`.

## Agents

The app is organized into 5 agents:

1. `agents/job_search_agent.py` — Agent 1 (job listing search)
2. `agents/job_scraper_agent.py` — Agent 2 (job description extraction)
3. `agents/profile_ingestion_agent.py` — Agent 3 (profile ingestion)
4. `agents/fit_analyzer_agent.py` — Agent 4 (fit analysis)
5. `agents/advisor_agent.py` — Agent 5 (career advisor + resume rewrite)

## Requirements

- Python 3.10+
- Anthropic API key

Install dependencies:

```bash
pip install -r requirements.txt
```

Install API dependencies (for FastAPI mode):

```bash
pip install fastapi uvicorn
```

## Environment Variables

Create `.env` in project root (or update existing):

```env
ANTHROPIC_API_KEY=your_key_here
```

Optional (if used by profile ingestion fallback paths):

```env
PROXYCURL_API_KEY=your_proxycurl_key
```

## Folder Notes

- `inputs/`: local input files (resume, CV, etc.)
- `outputs/`: generated outputs (JSON/CSV/MD/TXT)

## CLI Usage

All commands below are run from project root.

### 1) Profile Ingestion

```bash
py run_profile.py --resume "inputs/Resume.pdf" --github your-github-username
```

Other supported options:

```bash
py run_profile.py --cv "inputs/cv.docx" --linkedin "https://www.linkedin.com/in/your-profile"
py run_profile.py --text "I am targeting AI Engineer roles in Ireland."
```

Output:

- `outputs/output_profile.json`

### 2) Job Search (Agent 1 only)

```bash
py run_job_search.py --title "AI Engineer" --location "Remote" --days 14 --max 10
```

Output:

- `output_job_listings.json` (currently written to project root)

### 3) Search + Scrape (Agents 1 -> 2)

```bash
py run_scraper.py --title "AI Engineer" --location "Remote" --days 30 --max 10
```

Outputs:

- `outputs/output_job_title_urls.csv`
- `outputs/output_job_descriptions.json`

### 4) Fit Analyzer (Agent 4)

Mode A: from saved files

```bash
py run_fit_analyzer.py --jd outputs/output_job_descriptions.json --profile outputs/output_profile.json
```

Mode B: run Agents 1 -> 3 first, then fit analysis

```bash
py run_fit_analyzer.py --title "AI Engineer" --resume "inputs/Resume.pdf" --github your-github-username --max 10
```

Output:

- `outputs/output_fit_report.json`

### 5) Advisor (Agent 5)

Mode A: from saved files

```bash
py run_advisor.py --jd outputs/output_job_descriptions.json --profile outputs/output_profile.json --fit outputs/output_fit_report.json --out outputs
```

Mode B: full pipeline (Agents 1 -> 5)

```bash
py run_advisor.py --title "AI Engineer" --resume "inputs/Resume.pdf" --github your-github-username --out outputs
```

Outputs in `--out` directory (default: `output`):

- `resume_improved.md`
- `resume_improved.txt`
- `fit_report.md`
- `action_plan.md`
- `job_links.txt`
- `advisor_output.json`

## Typical End-to-End Flow (CLI)

Recommended sequence:

1. Profile: `py run_profile.py --resume "inputs/Resume.pdf" --github your-github-username`
2. Search/scrape: `py run_scraper.py --title "AI Engineer" --location "Remote" --max 10`
3. Fit: `py run_fit_analyzer.py --jd outputs/output_job_descriptions.json --profile outputs/output_profile.json`
4. Advisor: `py run_advisor.py --jd outputs/output_job_descriptions.json --profile outputs/output_profile.json --fit outputs/output_fit_report.json --out outputs`

## FastAPI Usage

Start API server:

```bash
uvicorn api_app:app --reload
```

Docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

### Endpoint Sections (Swagger Tags)

1. **App Health**
- `GET /health`

2. **Complete Advisor Run**
- `POST /run/career-advisor/full-pipeline`

3. **Individual Agent Runs**
- `POST /run/career-advisor`
- `POST /run/fit-analyzer`
- `POST /run/job-scraper`
- `POST /run/job-search`
- `POST /run/profile-ingester`

### Example API Calls

Health:

```bash
curl http://127.0.0.1:8000/health
```

Profile ingester:

```bash
curl -X POST http://127.0.0.1:8000/run/profile-ingester \
  -H "Content-Type: application/json" \
  -d '{
    "resume": "inputs/Resume.pdf",
    "github": "example-user",
    "linkedin": "https://www.linkedin.com/in/example"
  }'
```

Job search:

```bash
curl -X POST http://127.0.0.1:8000/run/job-search \
  -H "Content-Type: application/json" \
  -d '{
    "title": "AI Engineer",
    "location": "Remote",
    "days": 14,
    "max": 10
  }'
```

Job scraper:

```bash
curl -X POST http://127.0.0.1:8000/run/job-scraper \
  -H "Content-Type: application/json" \
  -d '{
    "title": "AI Engineer",
    "location": "Remote",
    "days": 30,
    "max": 8
  }'
```

Fit analyzer (saved-input mode):

```bash
curl -X POST http://127.0.0.1:8000/run/fit-analyzer \
  -H "Content-Type: application/json" \
  -d '{
    "jd": "outputs/output_job_descriptions.json",
    "profile": "outputs/output_profile.json",
    "max": 10
  }'
```

Career advisor (saved-input mode):

```bash
curl -X POST http://127.0.0.1:8000/run/career-advisor \
  -H "Content-Type: application/json" \
  -d '{
    "jd": "outputs/output_job_descriptions.json",
    "profile": "outputs/output_profile.json",
    "fit": "outputs/output_fit_report.json",
    "out": "outputs"
  }'
```

Career advisor full pipeline:

```bash
curl -X POST http://127.0.0.1:8000/run/career-advisor/full-pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "title": "AI Engineer",
    "location": "Remote",
    "resume": "inputs/Resume.pdf",
    "github": "example-user",
    "linkedin": "https://www.linkedin.com/in/example",
    "max": 10,
    "out": "outputs"
  }'
```

## API Response Format

All runner endpoints return:

```json
{
  "command": ["..."],
  "exit_code": 0,
  "stdout": "...",
  "stderr": "..."
}
```

`exit_code != 0` means the underlying script failed. Check `stderr` and `stdout` for details.

## Troubleshooting

1. **429 / rate limits from Anthropic**
- Reduce workload (`--max`), lower concurrency in scraper, or retry later.

2. **No module named ...**
- Reinstall dependencies:
  - `pip install -r requirements.txt`
  - `pip install fastapi uvicorn` (for API)

3. **Uvicorn exits quickly with `--reload`**
- This can happen during file-change reload cycles. Re-run:
  - `uvicorn api_app:app --reload`

4. **Empty or weak results**
- Verify API key in `.env`.
- Check that source files/URLs are valid and accessible.

## Project Structure (High-level)

- `agents/` — core agent logic
- `models/` — dataclasses/types
- `tools/` — web/file/helpers used by agents
- `utils/` — loaders and utility helpers
- `run_*.py` — CLI entry points
- `api_app.py` — FastAPI wrapper
- `outputs/` — generated outputs
