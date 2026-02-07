# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dash is a data agent that delivers **insights, not just SQL results**. Uses pure Anthropic SDK (no framework dependencies).

## Architecture Diagram

```mermaid
flowchart TB
    subgraph Frontend ["Next.js Frontend (dash-ui)"]
        UI[Chat Interface]
        ChartRenderer[Chart Renderer]
    end

    subgraph API ["FastAPI Server (api_server.py)"]
        SSE[SSE Streaming Endpoint]
        SessionMgr[Session Manager]
    end

    subgraph Agent ["DashAgent (dash_agent.py)"]
        Claude[Claude Opus 4.6 API]
        ToolLoop[Tool Execution Loop]
        History[Conversation History]

        subgraph Tools ["Available Tools"]
            RunSQL[run_sql]
            RunCode[run_code]
            RunChart[run_code_and_get_chart]
            ListVars[list_variables]
        end

        subgraph Interpreter ["Code Interpreter"]
            Globals[Persistent State]
            Pandas[pandas/numpy]
            Matplotlib[matplotlib]
        end
    end

    subgraph Data ["Data Layer"]
        Postgres[(PostgreSQL DB)]
        Knowledge[Knowledge Files]
    end

    UI -->|POST /chat/stream| SSE
    SSE -->|Events: tool_start, delta, tool_complete, done| UI
    ChartRenderer -->|Renders base64| UI

    SSE --> SessionMgr
    SessionMgr -->|Get/Create Agent| Agent

    Claude -->|Streaming Response| ToolLoop
    ToolLoop -->|Execute| Tools
    Tools --> Interpreter

    RunSQL -->|SQL Query| Postgres
    RunCode --> Globals
    RunChart --> Matplotlib
    Matplotlib -->|base64 PNG| RunChart

    Knowledge -->|System Prompt Context| Claude
    History -->|Conversation Memory| Claude
```

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant Agent
    participant Claude
    participant Tools
    participant DB

    User->>Frontend: "Who won most championships?"
    Frontend->>API: POST /chat/stream
    API->>Agent: agent.run(message)
    Agent->>Claude: messages.stream()

    loop Tool Loop (until stop_reason != tool_use)
        Claude-->>Agent: ToolCallStarted (streaming)
        Agent-->>API: yield ToolCallStarted
        API-->>Frontend: SSE: tool_start

        Claude->>Agent: tool_use block
        Agent->>Tools: execute tool
        Tools->>DB: SQL query
        DB-->>Tools: results
        Tools-->>Agent: result string

        Agent-->>API: yield ToolCallCompleted
        API-->>Frontend: SSE: tool_complete

        Agent->>Claude: tool_result message
    end

    Claude-->>Agent: TextDelta (streaming)
    Agent-->>API: yield TextDelta
    API-->>Frontend: SSE: delta

    Agent-->>API: yield StreamDone
    API-->>Frontend: SSE: done
    Frontend-->>User: Rendered response + charts
```

## Structure

```
dash/
├── dash_agent.py         # Pure Anthropic SDK agent (MAIN AGENT)
├── agents.py             # Legacy Agno-based agent (deprecated)
├── paths.py              # Path constants
├── knowledge/            # Knowledge files (tables, queries, business rules)
│   ├── tables/           # Table metadata JSON files
│   ├── queries/          # Validated SQL queries
│   └── business/         # Business rules and metrics
└── tools/
    └── code_interpreter.py # Legacy (now built into dash_agent.py)

api_server.py             # FastAPI server with SSE streaming
../dash-ui/               # Next.js frontend (separate directory)

db/
├── session.py            # PostgreSQL session factory
└── url.py                # Database URL builder
```

## Commands

```bash
# Setup
source .venv/bin/activate

# Run API Server
python api_server.py

# Run Frontend (separate terminal)
cd ../dash-ui && npm run dev
```

## Architecture (Pure Anthropic SDK)

```python
from dash.dash_agent import DashAgent
from db.url import db_url

agent = DashAgent(db_url=db_url)

# Streaming chat
for chunk in agent.chat("What tables are available?"):
    print(chunk, end="")

# Or sync
response = agent.chat_sync("How many drivers won championships?")
```

**Tools available to the agent:**
- `run_sql` - Execute SQL queries against PostgreSQL
- `run_code` - Execute Python code (persistent state like Jupyter)
- `run_code_and_get_chart` - Create matplotlib charts (agent controls styling)
- `list_variables` - Show variables in Python session

## Code Interpreter

Built into `dash_agent.py` with:
- **Persistent state**: Variables persist across tool calls
- **Pre-loaded**: pandas as `pd`, numpy as `np`, `run_sql(query)` function
- **Charts**: Returned as base64 with markers `[CHART_BASE64]...[/CHART_BASE64]`

## API Server

**Endpoints:**
- `GET /` - Health check
- `POST /chat` - Non-streaming chat
- `POST /chat/stream` - SSE streaming
- `POST /clear` - Clear session history

**Chart Handling:**
- Charts marked with `[CHART_BASE64]...[/CHART_BASE64]`
- Frontend extracts base64 and renders inline

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key (for Claude) |
| `DB_*` | No | Database config (defaults to localhost) |
