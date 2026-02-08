# Dash

Dash is a **data agent** that delivers insights, not just SQL results. Ask questions in plain English, get analysis with charts and explanations.

Built with the Anthropic SDK and Claude — no framework dependencies.

## Quick Start

```sh
# Clone and setup
git clone https://github.com/vibedatascience/dash_trial.git && cd dash_trial
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Add your Anthropic API key
cp example.env .env  # then add ANTHROPIC_API_KEY=sk-ant-***

# Start PostgreSQL
docker compose up -d dash-db

# Load sample data (F1 racing, 1950-2024)
python -m dash.scripts.load_data

# Run the API server
python api_server.py
```

The API runs at [http://localhost:8000](http://localhost:8000).

## Frontend

The frontend lives in a separate repo: [vibedatascience/dash-ui](https://github.com/vibedatascience/dash-ui)

```sh
cd ../dash-ui
npm install && npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## What It Can Do

Dash isn't limited to SQL — it's a general-purpose data agent:

- **Fetch data from URLs** — CSV, JSON, any public API
- **Query the database** — pre-loaded F1 dataset or your own PostgreSQL
- **Run Python or R** — pandas, numpy, matplotlib, tidyverse, ggplot2
- **Create charts** — matplotlib, ggplot2, or interactive D3.js
- **Scrape the web** — requests + pandas for web data

**Try it:**
- "Use yfinance to pull AMZN stock data and plot a 50-day moving average"
- "Fetch the latest TidyTuesday dataset and find interesting patterns"
- "Who won the most F1 championships? Show me a chart"
- "Fetch GDP data from the World Bank API for G7 countries"

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│   Next.js UI     │────▶│  FastAPI Server   │────▶│  DashAgent   │
│  (dash-ui repo)  │◀────│  (api_server.py)  │◀────│  (Claude)    │
│                  │ SSE │                   │     │              │
└──────────────────┘     └──────────────────┘     └──────┬───────┘
                                                         │
                                  ┌──────────────────────┼──────────────┐
                                  │                      │              │
                            ┌─────▼─────┐  ┌─────────▼──────┐  ┌──▼───────┐
                            │ PostgreSQL │  │ CodeInterpreter │  │ R Interp │
                            │ (F1 data) │  │ (Python/pandas) │  │ (ggplot) │
                            └───────────┘  └────────────────┘  └──────────┘
```

### Tools

| Tool | Purpose |
|------|---------|
| `run_sql` | Execute PostgreSQL queries |
| `run_code` | Execute Python with persistent state |
| `run_code_and_get_chart` | Create matplotlib/D3 charts |
| `run_r_code` | Execute R with persistent state |
| `run_r_chart` | Create ggplot2 charts |
| `list_variables` / `list_r_variables` | Inspect session state |

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chat/stream` | POST | SSE streaming chat |
| `/chat` | POST | Non-streaming chat |
| `/clear` | POST | Reset session |
| `/restore` | POST | Restore conversation history |
| `/conversations` | GET/POST | List or create conversations |
| `/conversations/{id}` | GET/PUT/DELETE | Manage a conversation |
| `/conversations/{id}/messages` | POST | Save messages |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `DB_HOST` | No | PostgreSQL host (default: localhost) |
| `DB_PORT` | No | PostgreSQL port (default: 5432) |
| `DB_USER` | No | PostgreSQL user (default: ai) |
| `DB_PASS` | No | PostgreSQL password (default: ai) |
| `DB_DATABASE` | No | Database name (default: ai) |

## Database

F1 racing data (1950-2024) is pre-loaded for demo purposes:

| Table | Rows | Description |
|-------|------|-------------|
| `drivers_championship` | ~1400 | Championship standings by year |
| `constructors_championship` | ~900 | Constructor standings by year |
| `race_results` | ~25000 | Individual race results |
| `race_wins` | ~1000 | Race winners |
| `fastest_laps` | ~1000 | Fastest lap records |

Conversations are stored in a `conversations` table (UUID, title, messages as JSONB).

## Adding Your Own Data

Point Dash at your own database by updating `DB_*` environment variables. Add context so the agent understands your schema:

```
dash/knowledge/
├── tables/      # Table schemas and gotchas (JSON)
├── queries/     # Validated SQL patterns
└── business/    # Metrics and business rules
```

## Local Development

```sh
source .venv/bin/activate
docker compose up -d dash-db
python api_server.py          # API on :8000
cd ../dash-ui && npm run dev  # UI on :3000
python -m dash                # CLI mode (no frontend needed)
```
