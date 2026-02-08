"""
Pure Anthropic SDK Data Agent
=============================

A data analysis agent using the Anthropic API directly (no Agno framework).
Handles conversation history, tool execution, and streaming responses.
"""

import json
import logging
import os
import re
import io
import base64
from dataclasses import dataclass
from typing import Any, Generator, Literal

import anthropic
import pandas as pd
from sqlalchemy import create_engine, text

from dash.paths import TABLES_DIR, BUSINESS_DIR
from dash.r_interpreter import RInterpreter, R_TOOLS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Event Types (matching what the frontend expects from old Agno)
# =============================================================================

@dataclass
class ToolCallStarted:
    """Emitted when a tool call starts."""
    event: Literal["ToolCallStarted"] = "ToolCallStarted"
    tool_name: str = ""
    tool_args: dict = None

    def __post_init__(self):
        if self.tool_args is None:
            self.tool_args = {}


@dataclass
class ToolCallCompleted:
    """Emitted when a tool call completes."""
    event: Literal["ToolCallCompleted"] = "ToolCallCompleted"
    tool_name: str = ""
    tool_args: dict = None
    result: str = ""

    def __post_init__(self):
        if self.tool_args is None:
            self.tool_args = {}


@dataclass
class TextDelta:
    """Emitted for streaming text content."""
    event: Literal["TextDelta"] = "TextDelta"
    content: str = ""


@dataclass
class StreamDone:
    """Emitted when streaming is complete."""
    event: Literal["Done"] = "Done"


# =============================================================================
# Context Builders (replacing Agno's context modules)
# =============================================================================

def load_table_metadata() -> list[dict[str, Any]]:
    """Load table metadata from JSON files."""
    tables: list[dict[str, Any]] = []
    if not TABLES_DIR.exists():
        return tables

    for filepath in sorted(TABLES_DIR.glob("*.json")):
        try:
            with open(filepath) as f:
                table = json.load(f)
            tables.append({
                "table_name": table["table_name"],
                "description": table.get("table_description", ""),
                "use_cases": table.get("use_cases", []),
                "data_quality_notes": table.get("data_quality_notes", [])[:5],
            })
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error(f"Failed to load {filepath}: {e}")
    return tables


def format_semantic_model() -> str:
    """Format semantic model for system prompt."""
    tables = load_table_metadata()
    lines: list[str] = []

    for table in tables:
        lines.append(f"### {table['table_name']}")
        if table.get("description"):
            lines.append(table["description"])
        if table.get("use_cases"):
            lines.append(f"**Use cases:** {', '.join(table['use_cases'])}")
        if table.get("data_quality_notes"):
            lines.append("**Data quality:**")
            for note in table["data_quality_notes"]:
                lines.append(f"  - {note}")
        lines.append("")

    return "\n".join(lines)


def load_business_rules() -> dict[str, list[Any]]:
    """Load business definitions from JSON files."""
    business: dict[str, list[Any]] = {"metrics": [], "business_rules": [], "common_gotchas": []}

    if not BUSINESS_DIR.exists():
        return business

    for filepath in sorted(BUSINESS_DIR.glob("*.json")):
        try:
            with open(filepath) as f:
                data = json.load(f)
            for key in business:
                if key in data:
                    business[key].extend(data[key])
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load {filepath}: {e}")

    return business


def format_business_context() -> str:
    """Build business context string for system prompt."""
    business = load_business_rules()
    lines: list[str] = []

    if business["metrics"]:
        lines.append("## METRICS\n")
        for m in business["metrics"]:
            lines.append(f"**{m.get('name', 'Unknown')}**: {m.get('definition', '')}")
            if m.get("table"):
                lines.append(f"  - Table: `{m['table']}`")
            if m.get("calculation"):
                lines.append(f"  - Calculation: {m['calculation']}")
            lines.append("")

    if business["business_rules"]:
        lines.append("## BUSINESS RULES\n")
        for rule in business["business_rules"]:
            lines.append(f"- {rule}")
        lines.append("")

    if business["common_gotchas"]:
        lines.append("## COMMON GOTCHAS\n")
        for g in business["common_gotchas"]:
            lines.append(f"**{g.get('issue', 'Unknown')}**")
            if g.get("tables_affected"):
                lines.append(f"  - Tables: {', '.join(g['tables_affected'])}")
            if g.get("solution"):
                lines.append(f"  - Solution: {g['solution']}")
            lines.append("")

    return "\n".join(lines)


# =============================================================================
# Code Interpreter (persistent state like Jupyter)
# =============================================================================

class CodeInterpreter:
    """Execute Python code with persistent state (like Jupyter cells)."""

    def __init__(self, db_url: str | None = None):
        self._globals: dict[str, Any] = {}
        self._db_url = db_url

        # Give access to Python builtins (enables import, print, len, etc.)
        import builtins
        self._globals["__builtins__"] = builtins

        # Pre-populate with common imports
        self._globals["pd"] = pd
        self._globals["np"] = __import__("numpy")

        # Add database connection if available
        if db_url:
            try:
                engine = create_engine(db_url)
                self._globals["engine"] = engine
                self._globals["run_sql"] = lambda q: pd.read_sql(q, engine)
                self._globals["query_db"] = lambda q: pd.read_sql(q, engine)
            except Exception as e:
                logger.warning(f"Could not create DB engine: {e}")

    def run_code(self, code: str) -> str:
        """Execute Python code and return output.

        Uses contextlib.redirect_stdout/stderr instead of monkey-patching
        sys.stdout globally, so concurrent sessions don't steal each other's output.
        """
        from io import StringIO
        from contextlib import redirect_stdout, redirect_stderr

        stdout_buf = StringIO()
        stderr_buf = StringIO()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, self._globals)
            output = stdout_buf.getvalue()
            errors = stderr_buf.getvalue()

            if errors:
                return f"Output:\n{output}\n\nWarnings/Errors:\n{errors}"
            return output if output else "Code executed successfully (no output)"

        except SyntaxError:
            # If it's a single expression, evaluate it
            try:
                result = eval(code, self._globals)
                if result is not None:
                    return str(result)
                return "Code executed successfully (no output)"
            except Exception as e:
                return f"Error: {type(e).__name__}: {e}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def run_code_and_get_chart(self, code: str) -> str:
        """Execute code that creates a matplotlib chart, return base64 image."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Make plt available in the execution context
        self._globals["plt"] = plt

        try:
            # Clear any existing figures
            plt.clf()
            plt.close('all')

            # Execute the code (agent controls all styling via prompt instructions)
            exec(code, self._globals)

            # Get the current figure
            fig = plt.gcf()

            # Save to base64 - no styling overrides, agent controls everything
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')

            plt.close('all')

            return f"[CHART_BASE64]{img_base64}[/CHART_BASE64]"

        except Exception as e:
            plt.close('all')
            return f"Chart Error: {type(e).__name__}: {e}"

    def list_variables(self) -> str:
        """List all user-defined variables in the current session."""
        user_vars = {
            k: type(v).__name__
            for k, v in self._globals.items()
            if not k.startswith('_') and k not in ['pd', 'np', 'plt', 'engine', 'run_sql', 'query_db']
        }
        if not user_vars:
            return "No user-defined variables in session."
        return "\n".join(f"{k}: {v}" for k, v in user_vars.items())


# =============================================================================
# Tool Definitions for Claude
# =============================================================================

TOOLS = [
    {
        "name": "run_sql",
        "description": """Execute a SQL query against the PostgreSQL database and return results.

Use this tool to:
- Query data from the available tables
- Join tables to answer complex questions
- Aggregate data for analysis

The results will be returned as a formatted table. Always use LIMIT for large result sets.
Available tables and their schemas are provided in the system prompt.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The SQL query to execute. Use PostgreSQL syntax."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "run_code",
        "description": """Execute Python code for data analysis and manipulation.

Use this tool to:
- Process and transform data from SQL queries
- Perform calculations and statistical analysis
- Prepare data for visualization

The execution environment persists between calls (like Jupyter cells).
Pre-loaded: pandas as pd, numpy as np, query_db(sql) function.

IMPORTANT: Store results in variables to use in later code cells.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use print() to show output."
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "run_code_and_get_chart",
        "description": """Execute Python code that creates a matplotlib visualization.

Use this tool when you need to CREATE A CHART/VISUALIZATION.
The chart will be styled with a dark theme automatically.

Pre-loaded: matplotlib.pyplot as plt, pandas as pd, numpy as np, query_db(sql).

Example:
```python
df = query_db("SELECT date, revenue FROM sales")
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['revenue'])
plt.title('Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Revenue')
```

The chart is returned as a base64-encoded image.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code that creates a matplotlib chart."
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "list_variables",
        "description": "List all variables currently defined in the Python session. Use this to see what data you have already loaded.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "create_d3_chart",
        "description": """Create an interactive D3.js chart. ONLY use when user explicitly asks for D3 or interactive charts.

First fetch data using run_sql or run_code and store in a variable (e.g., df).
Then call this tool with D3.js code that uses the 'data' variable.

The 'data' variable will be automatically injected as an array of objects.
For example, if df has columns ['year', 'value'], data will be:
[{year: 1950, value: 45}, {year: 1960, value: 52}, ...]

Your D3 code should:
- Select '#chart' as the container
- Use the 'data' variable directly
- Set explicit width/height (e.g., 800x500)

Example:
```javascript
const width = 800, height = 500;
const svg = d3.select('#chart').append('svg').attr('width', width).attr('height', height);
const x = d3.scaleLinear().domain(d3.extent(data, d => d.year)).range([50, width-20]);
const y = d3.scaleLinear().domain([0, d3.max(data, d => d.value)]).range([height-30, 20]);
svg.selectAll('circle').data(data).join('circle')
   .attr('cx', d => x(d.year)).attr('cy', d => y(d.value)).attr('r', 5);
```""",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "D3.js code. The 'data' variable contains the data as array of objects."
                },
                "data_variable": {
                    "type": "string",
                    "description": "Name of the Python variable containing the DataFrame to use (e.g., 'df')"
                }
            },
            "required": ["code", "data_variable"]
        }
    }
]


# =============================================================================
# System Prompt
# =============================================================================

def build_system_prompt() -> str:
    """Build the complete system prompt with context."""
    semantic_model = format_semantic_model()
    business_context = format_business_context()

    return f"""You are Dash, an AI data analyst. Your job is to analyze data, create charts, and provide insights.

## What You Do

You help users with ANY data task:
- Fetch data from URLs (CSV, JSON, APIs)
- Parse and analyze text/data the user pastes
- Create visualizations and charts
- Run statistical analysis
- Answer questions about data

**Do whatever the user asks.** Don't assume they want SQL — most users will give you URLs, paste data, or ask general questions.

## Data Sources (in order of likelihood)

1. **URLs** - User gives you a CSV/JSON URL → use `pd.read_csv(url)` or `pd.read_json(url)`
2. **Pasted data** - User pastes text/CSV → parse it with pandas
3. **Web scraping** - User wants data from a webpage → use requests + pandas
4. **Pre-loaded PostgreSQL** - ONLY if user explicitly asks about "the database" or "tables" (F1 racing data is loaded for demo)

## Tools

### Python (default):
- `run_code` - Execute Python. Pre-loaded: pandas, numpy, requests
- `run_code_and_get_chart` - Create matplotlib charts
- `run_sql` - Query PostgreSQL (ONLY when user asks for database/tables)
- `list_variables` - See what's in memory

### R (when user prefers R or asks for tidyverse/ggplot2):
- `run_r_code` - Execute R code. Pre-loaded: **tidyverse** (dplyr, tidyr, readr, ggplot2)
- `run_r_chart` - Create ggplot2 visualizations (dark theme auto-applied)
- Use tidyverse idioms: pipes (`%>%`), `mutate()`, `filter()`, `group_by()`, `summarize()`
- For charts: always use ggplot2 with `aes()`, `geom_*()`, proper labels

## Examples

User: "Analyze this CSV: https://example.com/data.csv"
→ Use `pd.read_csv('https://example.com/data.csv')`

User: "Here's my sales data: [pastes CSV]"
→ Parse it with `pd.read_csv(io.StringIO('''...'''))`

User: "What tables are in the database?"
→ NOW use `run_sql` to query the PostgreSQL

## Guidelines

- **Never use emojis**
- Variables persist across calls (like Jupyter)
- ONE chart per response unless asked for more
- Always explain insights, not just raw numbers

## Chart Aesthetics (IMPORTANT)

**NEVER use default colors.** Every chart must have a cohesive, intentional color scheme tied to the data's subject matter.

### ONE CHART ONLY
**Create exactly ONE chart per response.** Do NOT create subplots, multi-panel figures, or "4 charts in 1" unless the user explicitly asks for multiple charts or a comparison grid. When in doubt, pick the single most insightful visualization.

### Color Philosophy:
- **Match the domain**: Brazil soccer data → green (#009c3b) and yellow (#ffdf00). Ferrari data → red (#dc0000). Ocean/marine data → deep blues and teals.
- **Avoid clichés**: NO purple gradients on white. NO rainbow palettes. NO generic blue-orange.
- **Use white or light backgrounds by default** - clean, professional look. Only use dark backgrounds (#1e293b) if it fits the data theme (e.g., space data, nightlife).
- **Limit palette**: 2-4 colors max. Every color must have a reason.

### Examples:
| Data Subject | Primary Colors | Background |
|--------------|----------------|------------|
| Financial/Banking | Navy (#1e3a5f), Gold (#d4af37) | White |
| Healthcare | Teal (#0d9488), Soft blue (#7dd3fc) | White/light gray |
| Sports team | Use ACTUAL team colors | White |
| Environmental | Forest green (#166534), Earth brown (#78350f) | Light cream |
| Tech/SaaS | Electric blue (#3b82f6), Slate (#334155) | White |

### Implementation:
- Python: Use `plt.figure(facecolor='white')` and `ax.set_facecolor('white')` by default
- R/ggplot2: Use `theme(panel.background = element_rect(fill = 'white'))`
- Only deviate from white background when thematically appropriate

---

## DATABASE SCHEMA (only use if user asks about "database" or "tables")

{semantic_model}
{business_context}"""


# =============================================================================
# Main Agent Class
# =============================================================================

class DashAgent:
    """
    A data analysis agent using the Anthropic API directly.

    Manages conversation history, tool execution, and streaming responses.
    Yields events compatible with old Agno format for frontend compatibility.
    """

    def __init__(
        self,
        db_url: str | None = None,
        model: str = "claude-opus-4-6",
        max_tokens: int = 16384,  # Opus 4 supports up to 64K output
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = build_system_prompt()

        # Initialize code interpreters with DB connection
        self.interpreter = CodeInterpreter(db_url)
        self.r_interpreter = RInterpreter(db_url)

        # Combined tools list (Python + R)
        self._tools = TOOLS + R_TOOLS

        # Conversation history: list of {"role": "user"|"assistant", "content": ...}
        self.messages: list[dict[str, Any]] = []

        # Database connection for direct SQL
        self._db_url = db_url
        self._engine = None
        if db_url:
            try:
                self._engine = create_engine(db_url)
            except Exception as e:
                logger.warning(f"Could not create DB engine: {e}")

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        try:
            if tool_name == "run_sql":
                query = tool_input.get("query", "")
                if not self._engine:
                    return "Error: No database connection available"
                df = pd.read_sql(query, self._engine)
                # Store in interpreter for later use
                self.interpreter._globals["_last_query_result"] = df
                return df.to_string(max_rows=50, max_cols=20)

            elif tool_name == "run_code":
                code = tool_input.get("code", "")
                return self.interpreter.run_code(code)

            elif tool_name == "run_code_and_get_chart":
                code = tool_input.get("code", "")
                return self.interpreter.run_code_and_get_chart(code)

            elif tool_name == "list_variables":
                return self.interpreter.list_variables()

            elif tool_name == "create_d3_chart":
                code = tool_input.get("code", "")
                data_variable = tool_input.get("data_variable", "")

                # Get the data from the interpreter's globals
                if data_variable not in self.interpreter._globals:
                    return f"Error: Variable '{data_variable}' not found. Available: {list(self.interpreter._globals.keys())}"

                data = self.interpreter._globals[data_variable]

                # Convert to JSON-serializable format
                if hasattr(data, 'to_dict'):
                    # It's a DataFrame
                    json_data = data.to_dict(orient='records')
                elif isinstance(data, list):
                    json_data = data
                else:
                    return f"Error: Variable '{data_variable}' is not a DataFrame or list"

                # Return D3 chart marker with code and data
                chart_payload = json.dumps({"code": code, "data": json_data})
                return f"[D3_CHART]{chart_payload}[/D3_CHART]"

            # R tools
            elif tool_name == "run_r_code":
                code = tool_input.get("code", "")
                return self.r_interpreter.run_code(code)

            elif tool_name == "run_r_chart":
                code = tool_input.get("code", "")
                return self.r_interpreter.run_code_and_get_chart(code)

            elif tool_name == "list_r_variables":
                return self.r_interpreter.list_variables()

            else:
                return f"Error: Unknown tool '{tool_name}'"

        except Exception as e:
            return f"Error executing {tool_name}: {type(e).__name__}: {e}"

    def run(self, user_message: str, stream: bool = True, stream_events: bool = True) -> Generator:
        """
        Send a message and stream the response as events.

        Yields event objects matching the old Agno format:
        - ToolCallStarted: when a tool starts
        - ToolCallCompleted: when a tool finishes (with result)
        - TextDelta: for streaming text chunks
        - StreamDone: when complete

        Args:
            user_message: The user's message
            stream: Whether to stream (always True for now)
            stream_events: Whether to yield tool events (always True for compatibility)
        """
        # Add user message to history
        self.messages.append({"role": "user", "content": user_message})

        while True:
            # Track tool calls that started during streaming
            pending_tools: dict[int, dict] = {}  # index -> {name, input partial}

            # Make API call with streaming
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                tools=self._tools,
                messages=self.messages,
            ) as stream_response:
                # Stream content as it arrives
                for event in stream_response:
                    if hasattr(event, 'type'):
                        # Text delta - yield immediately
                        if event.type == 'content_block_delta':
                            if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                                yield TextDelta(content=event.delta.text)

                        # Content block start - check if it's a tool_use
                        elif event.type == 'content_block_start':
                            if hasattr(event, 'content_block'):
                                block = event.content_block
                                if hasattr(block, 'type') and block.type == 'tool_use':
                                    # Tool use is starting! Emit event immediately
                                    tool_name = getattr(block, 'name', 'unknown')
                                    yield ToolCallStarted(
                                        tool_name=tool_name,
                                        tool_args={}  # Args come later in deltas
                                    )
                                    # Track this tool for later
                                    pending_tools[event.index] = {
                                        'name': tool_name,
                                        'id': getattr(block, 'id', ''),
                                    }

                # Get the final message
                final_message = stream_response.get_final_message()

            # Add assistant response to history
            self.messages.append({
                "role": "assistant",
                "content": final_message.content
            })

            # Check if we need to handle tool calls
            if final_message.stop_reason == "tool_use":
                # Execute tools and emit completion events
                tool_results = []
                for block in final_message.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        tool_id = block.id

                        # Execute the tool
                        result = self._execute_tool(tool_name, tool_input)

                        # Emit ToolCallCompleted event (start was already emitted during streaming)
                        # Send FULL result to frontend (includes base64 chart data)
                        yield ToolCallCompleted(
                            tool_name=tool_name,
                            tool_args=tool_input,
                            result=result
                        )

                        # For conversation history, strip base64 to avoid context overflow
                        # Charts can be 100K+ chars which quickly exceeds 200K token limit
                        history_result = result
                        if '[CHART_BASE64]' in result:
                            import re
                            history_result = re.sub(
                                r'\[CHART_BASE64\].*?\[/CHART_BASE64\]',
                                '[CHART_BASE64][Chart generated successfully - base64 data stripped from history][/CHART_BASE64]',
                                result,
                                flags=re.DOTALL
                            )

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": history_result
                        })

                # Add tool results to conversation
                self.messages.append({
                    "role": "user",
                    "content": tool_results
                })

                # Continue the loop to get Claude's response to tool results
                continue

            # No more tool calls, we're done
            break

        yield StreamDone()

    def chat(self, user_message: str) -> Generator[str, None, None]:
        """
        Simple text-only streaming interface.
        Yields only text chunks (for simpler use cases).
        """
        for event in self.run(user_message):
            if isinstance(event, TextDelta):
                yield event.content

    def chat_sync(self, user_message: str) -> str:
        """
        Send a message and get the complete response (non-streaming).
        """
        response_parts = []
        for event in self.run(user_message):
            if isinstance(event, TextDelta):
                response_parts.append(event.content)
        return "".join(response_parts)

    def clear_history(self):
        """Clear conversation history."""
        self.messages = []

    def restore_history(self, frontend_messages: list[dict[str, Any]]):
        """Restore conversation history from frontend-format messages.

        Converts the frontend event-based format to simplified text messages
        so Claude has conversational context when resuming a conversation.
        Tool calls are included as text summaries with their results.
        """
        self.messages = []
        for msg in frontend_messages:
            if msg.get("role") == "user":
                self.messages.append({"role": "user", "content": msg.get("content", "")})
            elif msg.get("role") == "assistant":
                parts = []
                for event in (msg.get("events") or []):
                    if event.get("type") == "tool" and event.get("tool"):
                        tool = event["tool"]
                        name = tool.get("name", "unknown")
                        args = tool.get("args", {})
                        code_or_query = args.get("query") or args.get("code") or ""
                        if code_or_query:
                            parts.append(f"[Executed {name}]:\n{code_or_query}")
                        else:
                            parts.append(f"[Executed {name}]")
                        result = str(tool.get("result", ""))
                        if result and "[CHART_BASE64]" not in result and "[D3_CHART]" not in result:
                            truncated = result[:500]
                            if len(result) > 500:
                                truncated += "... (truncated)"
                            parts.append(f"Result: {truncated}")
                        elif "[CHART_BASE64]" in result:
                            parts.append("Result: [Chart generated successfully]")
                    elif event.get("type") == "text" and event.get("content"):
                        parts.append(event["content"])
                if not parts and msg.get("content"):
                    parts.append(msg["content"])
                if parts:
                    self.messages.append({"role": "assistant", "content": "\n".join(parts)})

    def get_history(self) -> list[dict[str, Any]]:
        """Get the current conversation history."""
        return self.messages.copy()


# =============================================================================
# Convenience function
# =============================================================================

def create_agent(db_url: str | None = None) -> DashAgent:
    """Create a new DashAgent instance."""
    return DashAgent(db_url=db_url)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    from db.url import db_url

    agent = create_agent(db_url)

    print("Testing agent with events...")
    print("-" * 50)

    for event in agent.run("How many rows are in the drivers_championship table?"):
        if isinstance(event, TextDelta):
            print(event.content, end="", flush=True)
        elif isinstance(event, ToolCallStarted):
            print(f"\n[Tool Started: {event.tool_name}]")
        elif isinstance(event, ToolCallCompleted):
            print(f"\n[Tool Completed: {event.tool_name}]")
            print(f"  Result: {event.result[:100]}...")
        elif isinstance(event, StreamDone):
            print("\n[Done]")
