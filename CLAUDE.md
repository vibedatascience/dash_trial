# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dash is a self-learning data agent that delivers **insights, not just SQL results**. It grounds SQL generation in 6 layers of context and improves automatically with every query. Inspired by [OpenAI's in-house data agent](https://openai.com/index/inside-our-in-house-data-agent/).

## Structure

```
dash/
├── agents.py             # Dash agents (dash, reasoning_dash)
├── paths.py              # Path constants
├── knowledge/            # Knowledge files (tables, queries, business rules)
│   ├── tables/           # Table metadata JSON files
│   ├── queries/          # Validated SQL queries
│   └── business/         # Business rules and metrics
├── context/
│   ├── semantic_model.py # Layer 1: Table usage
│   └── business_rules.py # Layer 2: Business rules
├── tools/
│   ├── introspect.py     # Layer 6: Runtime context
│   ├── save_query.py     # Save validated queries
│   └── code_interpreter.py # Python execution with chart generation
├── scripts/
│   ├── load_data.py      # Load F1 sample data
│   └── load_knowledge.py # Load knowledge files
└── evals/
    ├── test_cases.py     # Test cases with golden SQL
    ├── grader.py         # LLM-based response grader
    └── run_evals.py      # Run evaluations

api_server.py             # FastAPI server with SSE streaming
../dash-ui/               # Next.js frontend (separate directory)

app/
├── main.py               # API entry point (AgentOS)
└── config.yaml           # Agent configuration

db/
├── session.py            # PostgreSQL session factory
└── url.py                # Database URL builder
```

## Commands

```bash
./scripts/venv_setup.sh && source .venv/bin/activate
./scripts/format.sh      # Format code
./scripts/validate.sh    # Lint + type check
python -m dash           # CLI mode
python -m dash.agents    # Test mode (runs sample query)

# Data & Knowledge
python -m dash.scripts.load_data       # Load F1 sample data
python -m dash.scripts.load_knowledge  # Load knowledge into vector DB

# Evaluations
python -m dash.evals.run_evals              # Run all evals (string matching)
python -m dash.evals.run_evals -c basic     # Run specific category
python -m dash.evals.run_evals -v           # Verbose mode (show responses)
python -m dash.evals.run_evals -g           # Use LLM grader
python -m dash.evals.run_evals -r           # Compare against golden SQL results
python -m dash.evals.run_evals -g -r -v     # All modes combined
```

## Architecture

**Two Learning Systems:**

| System | What It Stores | How It Evolves |
|--------|---------------|----------------|
| **Knowledge** | Validated queries, table metadata, business rules | Curated by you + Dash |
| **Learnings** | Error patterns, type gotchas, discovered fixes | Managed by Learning Machine automatically |

```python
# KNOWLEDGE: Static, curated (table schemas, validated queries)
dash_knowledge = Knowledge(...)

# LEARNINGS: Dynamic, discovered (error patterns, gotchas)
dash_learnings = Knowledge(...)

dash = Agent(
    knowledge=dash_knowledge,
    search_knowledge=True,
    learning=LearningMachine(
        knowledge=dash_learnings,  # separate from static knowledge
        user_profile=UserProfileConfig(mode=LearningMode.AGENTIC),
        user_memory=UserMemoryConfig(mode=LearningMode.AGENTIC),
        learned_knowledge=LearnedKnowledgeConfig(mode=LearningMode.AGENTIC),
    ),
    # History settings (CRITICAL for conversational context)
    add_history_to_context=True,
    read_chat_history=True,
    read_tool_call_history=True,  # Includes tool calls in history!
    num_history_runs=5,
)
```

**Learning Machine provides:**
- `search_learnings` / `save_learning` tools
- `user_profile` - structured facts about user
- `user_memory` - unstructured observations

## The Six Layers of Context

| Layer | Source | Code |
|-------|--------|------|
| 1. Table Usage | `dash/knowledge/tables/*.json` | `dash/context/semantic_model.py` |
| 2. Business Rules | `dash/knowledge/business/*.json` | `dash/context/business_rules.py` |
| 3. Query Patterns | `dash/knowledge/queries/*.sql` | Loaded into knowledge base |
| 4. Institutional Knowledge | Exa MCP | `dash/agents.py` |
| 5. Learnings | Learning Machine | Separate knowledge base |
| 6. Runtime Context | `introspect_schema` | `dash/tools/introspect.py` |

## Code Interpreter Tool

The agent has a Python code interpreter (`dash/tools/code_interpreter.py`) that works like Jupyter notebook cells:

- **Persistent state**: Variables persist across tool calls within a session
- **Tools**: `run_code`, `run_code_and_get_chart`, `list_variables`
- **Available libraries**: pandas, numpy, matplotlib, seaborn
- **Database access**: `query_db(sql)` function returns DataFrames
- **Timeout**: 30 second limit per execution
- **Charts**: Returned as base64 with markers `[CHART_BASE64]...[/CHART_BASE64]`

The frontend extracts these markers and renders charts inline.

## Data Quality (F1 Dataset)

| Issue | Solution |
|-------|----------|
| `position` is TEXT in `drivers_championship` | Use `position = '1'` |
| `position` is INTEGER in `constructors_championship` | Use `position = 1` |
| `date` is TEXT in `race_wins` | Use `TO_DATE(date, 'DD Mon YYYY')` |

## Evaluation System

Three evaluation modes (can be combined):

| Mode | Flag | Description |
|------|------|-------------|
| String matching | (default) | Check if expected strings appear in response |
| LLM grader | `-g` | Use GPT to evaluate response quality |
| Result comparison | `-r` | Execute golden SQL and compare results |

Test cases use `TestCase` dataclass with optional `golden_sql` for validation.

## API Server & Frontend

The project has a custom API server (`api_server.py`) and a Next.js frontend (`../dash-ui/`).

**Running locally with frontend:**
```bash
# Terminal 1: API Server (port 8000)
source .venv/bin/activate && python api_server.py

# Terminal 2: Frontend (port 3000)
cd ../dash-ui && npm run dev
```

**API Endpoints:**
- `GET /` - Health check
- `POST /chat` - Non-streaming chat
- `POST /chat/stream` - SSE streaming with tool call visibility

**Critical: Streaming with Tool Events**

When using `dash.run()` for streaming, you MUST use `stream_events=True` to receive tool call events:
```python
# WRONG - only text deltas, no tool events
response = dash.run(message, stream=True)

# CORRECT - includes ToolCallStarted, ToolCallCompleted events
response = dash.run(message, stream=True, stream_events=True)
```

Event types from Agno: `ToolCallStarted`, `ToolCallCompleted`, `RunContent`, `RunCompleted`

**Chart Handling in API:**
- Charts are returned with `[CHART_BASE64]...[/CHART_BASE64]` markers
- API server does NOT truncate results containing chart markers (full base64 needed)
- Frontend extracts base64, displays inline with zoom capability

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key (for Claude model) |
| `EXA_API_KEY` | No | Exa for web research |
| `DB_*` | No | Database config |
