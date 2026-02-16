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



# =============================================================================
# Code Interpreter (persistent state like Jupyter)
# =============================================================================

class CodeInterpreter:
    """Execute Python code with persistent state (like Jupyter cells)."""

    def __init__(self):
        from dash.paths import CHARTS_DIR
        self._globals: dict[str, Any] = {}

        # Persistent dir for any files the agent's code creates (charts, CSVs, etc.)
        CHARTS_DIR.mkdir(exist_ok=True)
        self._temp_dir = str(CHARTS_DIR)

        # Give access to Python builtins (enables import, print, len, etc.)
        import builtins
        self._globals["__builtins__"] = builtins

        # Pre-populate with common imports
        self._globals["pd"] = pd
        self._globals["np"] = __import__("numpy")

    def run_code(self, code: str) -> str:
        """Execute Python code and return output.

        Uses contextlib.redirect_stdout/stderr instead of monkey-patching
        sys.stdout globally, so concurrent sessions don't steal each other's output.
        Runs in a temp directory so any files created don't clutter the project.
        """
        import os
        from io import StringIO
        from contextlib import redirect_stdout, redirect_stderr

        stdout_buf = StringIO()
        stderr_buf = StringIO()
        prev_dir = os.getcwd()

        try:
            os.chdir(self._temp_dir)
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
        finally:
            os.chdir(prev_dir)

    def run_code_and_get_chart(self, code: str) -> str:
        """Execute code that creates a matplotlib chart, return base64 image."""
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Make plt available in the execution context
        self._globals["plt"] = plt
        prev_dir = os.getcwd()

        try:
            os.chdir(self._temp_dir)

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
        finally:
            os.chdir(prev_dir)

    def list_variables(self) -> str:
        """List all user-defined variables in the current session."""
        user_vars = {
            k: type(v).__name__
            for k, v in self._globals.items()
            if not k.startswith('_') and k not in ['pd', 'np', 'plt']
        }
        if not user_vars:
            return "No user-defined variables in session."
        return "\n".join(f"{k}: {v}" for k, v in user_vars.items())


# =============================================================================
# Tool Definitions for Claude
# =============================================================================

TOOLS = [
    {
        "name": "run_code",
        "description": """Execute Python code for data analysis and manipulation.

Use this tool to:
- Process and transform data (URLs, CSVs, APIs, user-pasted data)
- Perform calculations and statistical analysis
- Prepare data for visualization

The execution environment persists between calls (like Jupyter cells).
Pre-loaded: pandas as pd, numpy as np.

IMPORTANT: Store results in variables to use in later code cells.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Store results in variables — avoid print() for large outputs."
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

Pre-loaded: matplotlib.pyplot as plt, pandas as pd, numpy as np.

Example:
```python
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

First fetch data using run_code and store in a variable (e.g., df).
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
    return """You are Dash, an AI data analyst. Your job is to analyze data, create charts, and provide insights.

## CRITICAL: Think First, Then Execute

**Before touching any tool, ALWAYS do this:**

1. **Break down the problem from first principles.** Restate what the user is asking in precise, granular terms. Decompose it into sub-questions.
2. **State your assumptions explicitly.** What are you assuming about the data, the metric definitions, the time range, the filters, the grouping?
3. **Ask the user to confirm before proceeding.** If there is ANY ambiguity — what column to use, how to define a metric, what time period, what filters to apply, how to handle nulls/edge cases — ASK. Do not guess. List your assumptions as bullet points and ask "Does this look right, or should I adjust anything?"
4. **Only after the user confirms (or if the request is truly unambiguous), execute.**

**Example:**
User: "Show me the top performing products"
You should respond:
"Before I pull this, let me clarify a few things:
- **'Top performing'** — are we measuring by revenue, units sold, or profit margin?
- **Time range** — all time, last year, last quarter?
- **Top N** — top 5? top 10?

Let me know and I'll run the analysis."

Do NOT just run a query and hope it's what they meant.

## CRITICAL: Narrate Before Every Tool Call

**Before every tool call, write a brief explanation of what you're about to do and why.** The user should always understand your reasoning and approach before seeing tool execution. Think out loud — explain your logic from first principles.

**Examples:**
- "The dataset has 50k rows across 15 years, so I'll start by checking the shape and column types to understand what we're working with."
- "Since you want year-over-year growth, I need to calculate the percentage change between consecutive years. I'll group by year first, sum the revenue, then use pct_change()."
- "To find the outliers, I'll compute the IQR and flag anything beyond 1.5x the interquartile range — this is more robust than just using standard deviations since your data looks skewed."

**Never silently fire off a tool call.** Always give the user context first, even if it's one sentence.

## CRITICAL: Your Text Response IS the Deliverable

**The user reads YOUR response text, not tool call output. Tool calls are invisible computation — your written response is the only thing that matters.**

**Rules:**
1. Do NOT use `print()` to dump tables, summaries, or analysis in tool calls. The user cannot easily read tool output — it's collapsed, truncated, and hard to parse.
2. Store results in variables. Use tool calls purely to compute, transform, and prepare data.
3. After ALL tool calls finish, YOU write a thorough, detailed response explaining everything you found. This is your main job.
4. NEVER say "See above", "As shown in the output", or "The results show" without actually restating the findings in your response text.

**Your response MUST include:**
- The key takeaway up front
- Specific numbers, rankings, comparisons — everything the user needs to understand the data WITHOUT looking at tool output
- Context and interpretation (why does this matter? what's surprising?)
- If there's a chart, explain what it shows and what patterns are visible

**Bad example:**
- Tool call: `print(df.describe())` ... `print(df.head(20))` ... `print(top_10)`
- Response: "Here's what I found in the data above."

**Good example:**
- Tool call: `summary = df.describe()` ... `top_10 = df.nlargest(10, 'revenue')`
- Response: "The US leads with $4.2B in revenue, nearly double the UK at $2.1B. Germany rounds out the top 3 at $1.8B. Interestingly, the top 3 markets account for 68% of total revenue, while the remaining 12 markets split the rest fairly evenly..."

## What You Do

You help users with ANY data task:
- Fetch data from URLs (CSV, JSON, APIs)
- Parse and analyze text/data the user pastes
- Create visualizations and charts
- Run statistical analysis
- Answer questions about data

**Do whatever the user asks.** Most users will give you URLs, paste data, or ask general questions.

## Data Sources (in order of likelihood)

1. **URLs** - User gives you a CSV/JSON URL → use `pd.read_csv(url)` or `pd.read_json(url)`
2. **Pasted data** - User pastes text/CSV → parse it with pandas
3. **Web scraping** - User wants data from a webpage → use requests + pandas
4. **Pre-loaded variables** - Kaggle/TidyTuesday datasets may already be loaded as DataFrames — use `list_variables` to check

## Tools

### Python (default):
- `run_code` - Execute Python. Pre-loaded: pandas, numpy, requests
- `run_code_and_get_chart` - Create matplotlib charts
- `list_variables` - See what's in memory

### R (when user prefers R or asks for tidyverse/ggplot2):
- `run_r_code` - Execute R code. Pre-loaded: **tidyverse** (dplyr, tidyr, readr, ggplot2)
- `run_r_chart` - Create ggplot2 visualizations 
- Use tidyverse idioms: pipes (`%>%`), `mutate()`, `filter()`, `group_by()`, `summarize()`
- For charts: always use ggplot2 with `aes()`, `geom_*()`, proper labels

### Web (built-in, server-side):
- `web_search` - Search the web for real-time information, current events, latest data. Use when you need up-to-date info beyond your training data.
- `web_fetch` - Fetch full content from a URL (web page or PDF). Use to read documentation, articles, datasets, or any URL the user provides or that you found via web_search. Only works with URLs that appeared in the conversation.

## Examples

User: "Analyze this CSV: https://example.com/data.csv"
→ Use `pd.read_csv('https://example.com/data.csv')`

User: "Here's my sales data: [pastes CSV]"
→ Parse it with `pd.read_csv(io.StringIO('''...'''))`

## Guidelines

- **Never use emojis**
- Variables persist across calls (like Jupyter)
- ONE chart per response unless asked for more
- Always explain insights, not just raw numbers
- Compute in tools, narrate in your response. The user reads your text, not tool output.

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

"""


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
        model: str = "claude-opus-4-6",
        max_tokens: int = 16384,  # Opus 4 supports up to 64K output
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = build_system_prompt()

        # Initialize code interpreters
        self.interpreter = CodeInterpreter()
        self.r_interpreter = RInterpreter()

        # Combined tools list (Python + R + server tools)
        self._tools = TOOLS + R_TOOLS + [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
            {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 5},
        ]

        # Conversation history: list of {"role": "user"|"assistant", "content": ...}
        self.messages: list[dict[str, Any]] = []

    @staticmethod
    def _summarize_server_tool_result(block) -> str:
        """Extract a human-readable summary from a server tool result block."""
        content = getattr(block, 'content', None)
        if content is None:
            return "(no results)"

        # web_search_tool_result: content is a list of web_search_result objects
        if isinstance(content, list):
            titles = []
            for item in content:
                item_type = getattr(item, 'type', '')
                if item_type == 'web_search_result':
                    title = getattr(item, 'title', '')
                    url = getattr(item, 'url', '')
                    titles.append(f"{title}\n{url}" if title else url)
                elif item_type == 'web_search_tool_result_error':
                    return f"Search error: {getattr(item, 'error_code', 'unknown')}"
            return "\n\n".join(titles) if titles else "(no results)"

        # web_fetch_tool_result: content has a nested structure
        content_type = getattr(content, 'type', '')
        if content_type == 'web_fetch_tool_error':
            return f"Fetch error: {getattr(content, 'error_code', 'unknown')}"
        if content_type == 'web_fetch_result':
            url = getattr(content, 'url', '')
            return f"Fetched: {url}"

        return str(content)[:500]

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        try:
            if tool_name == "run_code":
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
                extra_headers={"anthropic-beta": "web-fetch-2025-09-10"},
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
                                block_type = getattr(block, 'type', '')

                                if block_type == 'tool_use':
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

                                elif block_type == 'server_tool_use':
                                    # Server tool (web_search, web_fetch) starting
                                    tool_name = getattr(block, 'name', 'unknown')
                                    yield ToolCallStarted(
                                        tool_name=tool_name,
                                        tool_args={}
                                    )

                                elif block_type in ('web_search_tool_result', 'web_fetch_tool_result'):
                                    # Server tool completed — extract result summary
                                    result_text = self._summarize_server_tool_result(block)
                                    tool_name = 'web_search' if 'search' in block_type else 'web_fetch'
                                    yield ToolCallCompleted(
                                        tool_name=tool_name,
                                        tool_args={},
                                        result=result_text
                                    )

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
                    elif event.get("type") == "context" and event.get("content"):
                        parts.append(f"[{event.get('title', 'Context')}]:\n{event['content']}")
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

def create_agent() -> DashAgent:
    """Create a new DashAgent instance."""
    return DashAgent()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    agent = create_agent()

    print("Testing agent with events...")
    print("-" * 50)

    for event in agent.run("What's 2 + 2? Use run_code to compute it."):
        if isinstance(event, TextDelta):
            print(event.content, end="", flush=True)
        elif isinstance(event, ToolCallStarted):
            print(f"\n[Tool Started: {event.tool_name}]")
        elif isinstance(event, ToolCallCompleted):
            print(f"\n[Tool Completed: {event.tool_name}]")
            print(f"  Result: {event.result[:100]}...")
        elif isinstance(event, StreamDone):
            print("\n[Done]")
