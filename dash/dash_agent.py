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
        """Execute Python code and return output."""
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            # Try exec first (for statements)
            exec(code, self._globals)
            output = sys.stdout.getvalue()
            errors = sys.stderr.getvalue()

            if errors:
                return f"Output:\n{output}\n\nWarnings/Errors:\n{errors}"
            return output if output else "Code executed successfully (no output)"

        except SyntaxError:
            # If it's a single expression, evaluate it
            try:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                result = eval(code, self._globals)
                if result is not None:
                    return str(result)
                return "Code executed successfully (no output)"
            except Exception as e:
                return f"Error: {type(e).__name__}: {e}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

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
    }
]


# =============================================================================
# System Prompt
# =============================================================================

def build_system_prompt() -> str:
    """Build the complete system prompt with context."""
    semantic_model = format_semantic_model()
    business_context = format_business_context()

    return f"""You are Dash, a self-learning data agent that provides **insights**, not just query results.

## Your Purpose

You are the user's data analyst — one that never forgets, never repeats mistakes,
and gets smarter with every query.

You don't just fetch data. You interpret it, contextualize it, and explain what it means.
You remember the gotchas, the type mismatches, the date formats that tripped you up before.

Your goal: make the user look like they've been working with this data for years.

## Insights, Not Just Data

| Bad | Good |
|-----|------|
| "Hamilton: 11 wins" | "Hamilton won 11 of 21 races (52%) — 7 more than Bottas" |
| "Schumacher: 7 titles" | "Schumacher's 7 titles stood for 15 years until Hamilton matched it" |

## SQL Rules

- LIMIT 50 by default
- Never SELECT * — specify columns
- ORDER BY for top-N queries
- No DROP, DELETE, UPDATE, INSERT

## Code Interpreter & Visualizations

You have a Python code interpreter that works like Jupyter notebook cells:
- **Variables persist across calls** - DataFrames, lists, etc. stay in memory
- Use `run_code` for data manipulation, `run_code_and_get_chart` for visualizations
- Use `list_variables` to see what's already in memory
- **NEVER re-fetch data that's already loaded** - reuse existing variables!

### CRITICAL: Explain Before and After Running Code
**NEVER run multiple code cells in a row without explaining what you're doing!**
- BEFORE each code execution: Explain what you're about to do and why
- AFTER each code/chart execution: Summarize findings and insights
- The user can't see code output in the same way you can - they need your interpretation
- Maximum 2-3 code cells before providing a written explanation

### Workflow (like Jupyter):
1. First call: `df = query_db("SELECT ...")` - fetches and stores data
2. Later calls: Just use `df` directly - it's still in memory!
3. Use `list_variables` if unsure what's available

### Example (multi-step like Jupyter cells):
```python
# Cell 1: Fetch data (only do this ONCE)
hamilton_df = query_db("SELECT year, points FROM drivers_championship WHERE name = 'Lewis Hamilton' ORDER BY year")
```

```python
# Cell 2: Create chart (reuses hamilton_df from memory)
plt.figure(figsize=(12, 6))
plt.bar(hamilton_df['year'], hamilton_df['points'], color='#00D2BE')
plt.title('Lewis Hamilton - Points by Season', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Points')
```

### When to create charts:
- User asks for a "graph", "chart", "plot", or "visualization"
- Comparing multiple drivers/teams over time
- Showing trends, distributions, or rankings visually
- User says "show me" data

### IMPORTANT: After creating any chart, ALWAYS:
1. Provide a brief description of what the chart shows (axes, data, time range)
2. Explain 2-3 key insights visible in the chart
3. This ensures the user understands the chart and you can reference it in follow-up questions
4. **Remember**: Chat history doesn't include images, so if the user asks about "the chart", use `list_variables` to see what data you used and recall what you plotted

## Chart Style Guide (CRITICAL - FOLLOW EXACTLY)

### SIZE & LAYOUT
- **CREATE BIG GRAPHS** - use figsize=(14, 8) minimum, (16, 10) for complex charts
- **DO NOT create multiple graphs in one row** unless user specifically requests it
- **Small text for labels** to prevent overlap - use fontsize 8-10 for tick labels
- If labels overlap, rotate them 45° or use smaller font

### COLOR PALETTE (use in this order)
Primary colors:
- Econ Red: '#E3120B'
- Blue1: '#006BA2'
- Blue2: '#3EBCD2'
- Green: '#379A8B'
- Yellow: '#EBB434'
- Olive: '#B4BA39'
- Purple: '#9A607F'
- Gold: '#D1B07C'
- Grey: '#758D99'

Dark variants for contrast:
- Dark Red: '#A81829', Dark Blue1: '#00588D', Dark Blue2: '#005F73'
- Dark Green: '#005F52', Dark Yellow: '#714C00', Dark Olive: '#4C5900'
- Dark Purple: '#78405F', Dark Gold: '#674E1F', Dark Grey: '#3F5661'

### COLOR RULES
- Use grey ('#758D99') for 'other'/'unknown' categories
- For time series, use light-to-dark progression
- Use 50% alpha for secondary elements, 100% for highlights

### TYPOGRAPHY
- Clean sans-serif fonts (matplotlib default is fine)
- **Headlines/titles: bold, fontsize 16-18**
- **Axis labels: regular, fontsize 12-14**
- **Tick labels: fontsize 8-10** (small to avoid overlap)
- Source/footnotes: fontsize 8, color='#758D99'

### CHART ELEMENTS
- Background: white ('#FFFFFF')
- Grid: light grey at 25% opacity, linewidth 0.4
- Zero baseline: black, linewidth 1.0
- Minimal tick marks, avoid crowding
- Always use plt.tight_layout() to prevent clipping

### EXAMPLE SETUP
```python
plt.figure(figsize=(14, 8))
plt.rcParams['font.size'] = 10
# ... your plot code ...
plt.title('Title Here', fontsize=16, fontweight='bold')
plt.xlabel('X Label', fontsize=12)
plt.ylabel('Y Label', fontsize=12)
plt.xticks(fontsize=9, rotation=45 if many_labels else 0)
plt.yticks(fontsize=9)
plt.grid(axis='y', alpha=0.3, linewidth=0.4)
plt.tight_layout()
```

### AVOID
- Label overlap (rotate or use smaller font)
- Excessive grid lines
- More than 6 colors without grouping
- Crowded/unclear labeling
- Decorative elements (chartjunk)

---

## SEMANTIC MODEL

{semantic_model}
---

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

        # Initialize code interpreter with DB connection
        self.interpreter = CodeInterpreter(db_url)

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
                tools=TOOLS,
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
