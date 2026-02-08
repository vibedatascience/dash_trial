# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dash is a data agent that delivers **insights, not just SQL results**. Built with pure Anthropic SDK - no framework dependencies.

## System Architecture

```mermaid
flowchart TB
    subgraph Frontend ["Next.js Frontend (dash-ui/)"]
        UI[Chat Interface<br/>app/page.js]
        ChartRenderer[Chart Renderer<br/>extracts base64]
    end

    subgraph API ["FastAPI Server (api_server.py)"]
        SSE["/chat/stream"<br/>SSE Streaming]
        REST["/chat"<br/>Non-streaming]
        Clear["/clear"<br/>Reset session]
        SessionMgr[Session Manager<br/>agents dict]
    end

    subgraph Agent ["DashAgent (dash/dash_agent.py)"]
        Claude[Claude API<br/>claude-opus-4-6]
        ToolLoop[Tool Execution Loop<br/>while stop_reason == tool_use]
        History[Conversation History<br/>self.messages]
        SystemPrompt[System Prompt<br/>semantic model + business rules]

        subgraph Tools ["Tools (TOOLS list)"]
            RunSQL["run_sql<br/>→ PostgreSQL"]
            RunCode["run_code<br/>→ exec()"]
            RunChart["run_code_and_get_chart<br/>→ matplotlib"]
            ListVars["list_variables<br/>→ session state"]
        end

        subgraph Interpreter ["CodeInterpreter class"]
            Globals["_globals dict<br/>(persistent state)"]
            PreLoaded["pd, np, plt<br/>run_sql(), query_db()"]
        end
    end

    subgraph Data ["Data Layer"]
        Postgres[(PostgreSQL<br/>F1 dataset)]
        Knowledge["Knowledge Files<br/>dash/knowledge/"]
    end

    UI -->|"POST /chat/stream<br/>{message, session_id}"| SSE
    SSE -->|"SSE events:<br/>tool_start, delta,<br/>tool_complete, done"| UI
    ChartRenderer -->|"Renders base64 PNG"| UI

    REST --> SessionMgr
    SSE --> SessionMgr
    SessionMgr -->|"get_agent(session_id)"| Agent

    SystemPrompt -->|"Injected at init"| Claude
    Claude -->|"messages.stream()"| ToolLoop
    ToolLoop -->|"_execute_tool()"| Tools
    Tools --> Interpreter
    Interpreter --> Globals

    RunSQL -->|"pd.read_sql()"| Postgres
    RunChart -->|"fig.savefig() → base64"| UI

    Knowledge -->|"load_table_metadata()<br/>load_business_rules()"| SystemPrompt
    History -->|"Appended each turn"| Claude
```

## Data Flow (Streaming)

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Next.js UI
    participant API as FastAPI
    participant Agent as DashAgent
    participant Claude as Claude API
    participant Tools
    participant DB as PostgreSQL

    User->>Frontend: "Who won most championships?"
    Frontend->>API: POST /chat/stream {message}
    API->>Agent: agent.run(message)
    Agent->>Agent: messages.append(user_message)
    Agent->>Claude: client.messages.stream()

    loop Tool Loop (until stop_reason != "tool_use")
        Claude-->>Agent: content_block_start (tool_use)
        Agent-->>API: yield ToolCallStarted
        API-->>Frontend: SSE: {"type": "tool_start", "name": "run_sql"}

        Claude->>Agent: tool_use block complete
        Agent->>Tools: _execute_tool("run_sql", {query})
        Tools->>DB: pd.read_sql(query)
        DB-->>Tools: DataFrame
        Tools-->>Agent: df.to_string()

        Agent-->>API: yield ToolCallCompleted
        API-->>Frontend: SSE: {"type": "tool_complete", "result": "..."}

        Agent->>Agent: messages.append(tool_result)
        Agent->>Claude: Continue with tool results
    end

    Claude-->>Agent: TextDelta (streaming content)
    Agent-->>API: yield TextDelta
    API-->>Frontend: SSE: {"type": "delta", "content": "..."}

    Agent-->>API: yield StreamDone
    API-->>Frontend: SSE: {"type": "done"}
    Frontend-->>User: Rendered markdown + charts
```

## Directory Structure

```
dash-repo/
├── api_server.py                 # FastAPI server — MAIN ENTRY POINT
│                                 # SSE streaming, conversations CRUD, session mgmt
├── dash/
│   ├── __init__.py               # Exports: DashAgent, create_agent
│   ├── __main__.py               # CLI: python -m dash
│   ├── dash_agent.py             # Core agent — DashAgent, CodeInterpreter, TOOLS, events
│   ├── r_interpreter.py          # R code execution (RInterpreter class)
│   ├── paths.py                  # Path constants (KNOWLEDGE_DIR, etc.)
│   ├── context/                  # System prompt builders
│   │   ├── semantic_model.py     # load_table_metadata()
│   │   └── business_rules.py     # load_business_rules()
│   ├── knowledge/                # Static knowledge (JSON/SQL files)
│   │   ├── tables/*.json         # Table schemas — {table_name, description, columns, gotchas}
│   │   ├── business/metrics.json # Metrics, business rules, common gotchas
│   │   └── queries/common_queries.sql
│   └── scripts/
│       └── load_data.py          # Downloads F1 CSVs → PostgreSQL
├── db/
│   ├── __init__.py               # Exports: db_url
│   └── url.py                    # Builds PostgreSQL connection URL
├── scripts/                      # Dev/deploy shell scripts
├── compose.yaml                  # Docker: PostgreSQL
├── Dockerfile                    # Container build
├── pyproject.toml                # Python project config
└── requirements.txt              # Dependencies
```

## Key Components

### DashAgent (dash/dash_agent.py)

```python
class DashAgent:
    def __init__(self, db_url: str, model: str = "claude-opus-4-6"):
        self.client = anthropic.Anthropic()
        self.interpreter = CodeInterpreter(db_url)  # Persistent Python state
        self.messages = []  # Conversation history

    def run(self, message: str) -> Generator[Event]:
        # Yields: ToolCallStarted, ToolCallCompleted, TextDelta, StreamDone

    def chat(self, message: str) -> Generator[str]:
        # Text-only streaming

    def chat_sync(self, message: str) -> str:
        # Non-streaming, returns full response
```

### Tools

| Tool | Purpose | Returns |
|------|---------|---------|
| `run_sql` | Execute PostgreSQL query | DataFrame as string |
| `run_code` | Execute Python (persistent state) | stdout or result |
| `run_code_and_get_chart` | Create matplotlib chart | `[CHART_BASE64]...[/CHART_BASE64]` |
| `list_variables` | Show session variables | Variable names + types |
| `run_r_code` | Execute R code (persistent state) | stdout or result |
| `run_r_chart` | Create ggplot2 chart | `[CHART_BASE64]...[/CHART_BASE64]` |
| `list_r_variables` | Show R session variables | Variable names + types |

### R Interpreter (dash/r_interpreter.py)

- **Persistent state**: R environment survives across tool calls via `.RData` files
- **Pre-loaded**: `dplyr`, `ggplot2`, `tidyr`, `query_db()`, `run_sql()`
- **Dark theme**: ggplot2 charts automatically styled for dark UI
- **Requires**: R and Rscript installed on system

### CodeInterpreter

- **Persistent state**: `_globals` dict survives across tool calls (like Jupyter)
- **Pre-loaded**: `pd`, `np`, `plt`, `run_sql()`, `query_db()`
- **Charts**: Saved to base64 PNG with markers for frontend extraction

## Commands

```bash
# Setup
source .venv/bin/activate

# Start PostgreSQL
docker compose up -d dash-db

# Load F1 data (first time only)
python -m dash.scripts.load_data

# Run API server (port 8000)
python api_server.py

# Run frontend (port 3000, separate terminal)
cd ../dash-ui && npm run dev

# CLI mode (interactive)
python -m dash
```

## API Endpoints

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/` | GET | - | `{"status": "ok", "model": "..."}` |
| `/chat` | POST | `{message, session_id?, language?}` | `{response, tool_calls, charts}` |
| `/chat/stream` | POST | `{message, session_id?, language?}` | SSE stream |
| `/clear` | POST | `{session_id?}` | `{"status": "cleared"}` |
| `/restore` | POST | `{session_id, messages}` | `{"status": "restored"}` |
| `/conversations` | GET | - | `{conversations: [...]}` |
| `/conversations` | POST | `{title?, language?}` | `{id, status}` |
| `/conversations/{id}` | GET | - | `{id, title, messages, language, ...}` |
| `/conversations/{id}` | PUT | `{title?, messages?}` | `{status, id}` |
| `/conversations/{id}` | DELETE | - | `{status, id}` |
| `/conversations/{id}/messages` | POST | `{messages}` | `{status, id}` |

### SSE Event Types

```json
{"type": "tool_start", "name": "run_sql", "args": {...}}
{"type": "tool_complete", "name": "run_sql", "result": "..."}
{"type": "delta", "content": "Hamilton won..."}
{"type": "done"}
{"type": "error", "error": "..."}
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `DB_HOST` | No | PostgreSQL host (default: localhost) |
| `DB_PORT` | No | PostgreSQL port (default: 5432) |
| `DB_USER` | No | PostgreSQL user (default: ai) |
| `DB_PASS` | No | PostgreSQL password (default: ai) |
| `DB_DATABASE` | No | Database name (default: ai) |

## Database Schema

### F1 Dataset Tables

| Table | Rows | Key Columns | Gotchas |
|-------|------|-------------|---------|
| `drivers_championship` | ~1400 | year, position, name, team, points | position is TEXT ('1', '2', 'Ret') |
| `constructors_championship` | ~900 | year, position, team, points | position is INTEGER (different!) |
| `race_results` | ~25000 | year, position, name, team, venue | position is TEXT |
| `race_wins` | ~1000 | date, name, team, venue | date is TEXT ('DD Mon YYYY') |
| `fastest_laps` | ~1000 | year, name, team, venue, time | - |

### Conversation Storage

| Table | Columns |
|-------|---------|
| `conversations` | id (UUID), title, messages (JSONB), language, created_at, updated_at |
