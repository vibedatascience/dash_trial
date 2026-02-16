"""
Simple API server for Dash with streaming
Run: python api_server.py

Uses pure Anthropic SDK (no Agno framework dependency)
"""

import os
import json
import asyncio
import sys
import re
import time
import uuid
import threading
import httpx
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine, text
from datetime import datetime

# API keys should be set via environment variables
# ANTHROPIC_API_KEY - for Claude model

import anthropic

from dash.dash_agent import (
    DashAgent,
    ToolCallStarted,
    ToolCallCompleted,
    TextDelta,
    StreamDone
)
from db.url import db_url

# Lightweight client for title generation, chart fixes, and suggestions
_title_client = anthropic.Anthropic()


def _generate_suggestions(context: str, title: str = "", description: str = "") -> list[str]:
    """Generate 3 follow-up suggestions using Haiku based on conversation context."""
    try:
        header = f"Dataset: {title}\n" if title else ""
        header += f"Description: {description}\n" if description else ""

        prompt = f"""{header}Context:
{context[:2000]}

Suggest exactly 3 short follow-up questions or analyses the user could try next. Be specific — reference actual data, columns, or findings from the context. Under 80 chars each. One per line. No numbering, no bullets, no quotes."""

        resp = _title_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
            system="You suggest data analysis follow-ups. Output exactly 3 lines, nothing else.",
        )
        lines = [l.strip() for l in resp.content[0].text.strip().split('\n') if l.strip()]
        return lines[:4]
    except Exception:
        return []

# Vision sub-agent system prompt for fixing chart layout issues
CHART_FIX_SYSTEM_PROMPT = """You are a chart layout specialist. You receive a chart image and the code that generated it. Your ONLY job is to fix VISUAL and LAYOUT issues (overlapping labels, clipped text, cramped legends, dense tick marks, small fonts, poor spacing).

CRITICAL RULES — violating any of these is a failure:
1. DO NOT rename, remove, or redefine ANY variable. Every variable name (DataFrames, series, lists, etc.) must stay EXACTLY as in the original code. The code runs in a persistent session where those variables already exist.
2. DO NOT change data, filtering, grouping, aggregation, chart type, color palette, or theme.
3. Make the MINIMUM changes necessary — only add/modify layout parameters (figsize, tight_layout, rotation, fontsize, margins, legend position, label wrapping).
4. Return ONLY the corrected code in a single fenced code block. No explanation before or after.

Common fixes: plt.xticks(rotation=45, ha='right'), plt.tight_layout(), larger figsize, plt.subplots_adjust(), legend(fontsize=..., loc=...), wrapping long labels with textwrap, reducing tick density with MaxNLocator."""

app = FastAPI(title="Dash API")
executor = ThreadPoolExecutor(max_workers=8)

# Database engine for conversations
db_engine = create_engine(db_url)

# Session expiration time (30 minutes)
SESSION_EXPIRY_SECONDS = 30 * 60

# Store agents by session_id for conversation persistence
# Format: {session_id: {"agent": DashAgent, "last_access": timestamp, "lock": threading.Lock()}}
sessions: dict[str, dict] = {}
sessions_dict_lock = threading.Lock()  # Protects creation/deletion of session entries


def debug_log(msg):
    """Print debug message to stderr"""
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)


def cleanup_expired_sessions():
    """Remove sessions that haven't been accessed in SESSION_EXPIRY_SECONDS."""
    now = time.time()
    expired = [
        sid for sid, data in sessions.items()
        if now - data["last_access"] > SESSION_EXPIRY_SECONDS
    ]
    for sid in expired:
        debug_log(f"Cleaning up expired session: {sid}")
        # Clean up R interpreter temp files if present
        agent = sessions[sid]["agent"]
        if hasattr(agent, 'r_interpreter') and agent.r_interpreter:
            agent.r_interpreter.cleanup()
        del sessions[sid]
    if expired:
        debug_log(f"Cleaned up {len(expired)} expired sessions")


def get_session(session_id: str | None) -> dict:
    """Get or create a session (agent + lock) for the session_id."""
    if session_id is None:
        session_id = "default"

    with sessions_dict_lock:
        # Clean up expired sessions periodically
        cleanup_expired_sessions()

        now = time.time()
        if session_id not in sessions:
            debug_log(f"Creating new agent for session: {session_id}")
            sessions[session_id] = {
                "agent": DashAgent(),
                "lock": threading.Lock(),
                "last_access": now,
            }
        else:
            sessions[session_id]["last_access"] = now

    return sessions[session_id]


def get_agent(session_id: str | None) -> DashAgent:
    """Get or create an agent for the session."""
    return get_session(session_id)["agent"]


# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    conversation_id: str | None = None  # UUID for persistent storage
    language: str | None = "python"  # "python" or "r"


class KaggleLoadRequest(BaseModel):
    ref: str            # e.g. "saidaminsaidaxmadov/chocolate-sales"
    title: str
    subtitle: str | None = None
    totalBytes: int | None = None
    language: str | None = "python"


class TidyTuesdayLoadRequest(BaseModel):
    date: str           # e.g. "2026-02-04"
    year: int           # e.g. 2026
    title: str
    source: str | None = None
    article: str | None = None
    language: str | None = "python"


class OwidLoadRequest(BaseModel):
    slug: str           # e.g. "life-expectancy"
    title: str
    topic: str | None = None
    language: str | None = "python"

class FredLoadRequest(BaseModel):
    series_id: str      # e.g. "GDP"
    title: str
    category: str | None = None
    language: str | None = "python"

class WorldBankLoadRequest(BaseModel):
    indicator_id: str   # e.g. "NY.GDP.MKTP.CD"
    title: str
    category: str | None = None
    language: str | None = "python"


class RestoreRequest(BaseModel):
    session_id: str | None = None
    messages: list = []


class ChartFixRequest(BaseModel):
    session_id: str
    code: str
    chart_base64: str
    tool_name: str  # "run_code_and_get_chart" or "run_r_chart"


class ConversationCreate(BaseModel):
    title: str | None = None
    language: str | None = "python"


class ConversationUpdate(BaseModel):
    title: str | None = None
    messages: list | None = None


class Conversation(BaseModel):
    id: str
    title: str | None
    messages: list
    language: str
    created_at: str
    updated_at: str


# ============================================================
# Conversation Database Functions
# ============================================================

def create_conversation(title: str | None = None, language: str = "python") -> str:
    """Create a new conversation and return its ID."""
    with db_engine.connect() as conn:
        result = conn.execute(
            text("""
                INSERT INTO conversations (title, language)
                VALUES (:title, :language)
                RETURNING id
            """),
            {"title": title, "language": language}
        )
        conn.commit()
        row = result.fetchone()
        return str(row[0])


def get_conversation(conversation_id: str) -> dict | None:
    """Get a conversation by ID."""
    with db_engine.connect() as conn:
        result = conn.execute(
            text("SELECT id, title, messages, language, created_at, updated_at FROM conversations WHERE id = :id"),
            {"id": conversation_id}
        )
        row = result.fetchone()
        if row:
            return {
                "id": str(row[0]),
                "title": row[1],
                "messages": row[2] if row[2] else [],
                "language": row[3],
                "created_at": row[4].isoformat() if row[4] else None,
                "updated_at": row[5].isoformat() if row[5] else None,
            }
        return None


def list_conversations(limit: int = 50) -> list[dict]:
    """List recent conversations."""
    with db_engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT id, title, messages, language, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT :limit
            """),
            {"limit": limit}
        )
        conversations = []
        for row in result:
            # Extract first user message for preview if no title
            messages = row[2] if row[2] else []
            title = row[1]
            if not title and messages:
                for msg in messages:
                    if msg.get("role") == "user":
                        title = msg.get("content", "")[:50]
                        if len(msg.get("content", "")) > 50:
                            title += "..."
                        break

            conversations.append({
                "id": str(row[0]),
                "title": title or "New conversation",
                "message_count": len(messages),
                "language": row[3],
                "created_at": row[4].isoformat() if row[4] else None,
                "updated_at": row[5].isoformat() if row[5] else None,
            })
        return conversations


def update_conversation_messages(conversation_id: str, messages: list) -> bool:
    """Update conversation messages."""
    with db_engine.connect() as conn:
        result = conn.execute(
            text("""
                UPDATE conversations
                SET messages = :messages, updated_at = NOW()
                WHERE id = :id
            """),
            {"id": conversation_id, "messages": json.dumps(messages)}
        )
        conn.commit()
        return result.rowcount > 0


def update_conversation_title(conversation_id: str, title: str) -> bool:
    """Update conversation title."""
    with db_engine.connect() as conn:
        result = conn.execute(
            text("""
                UPDATE conversations
                SET title = :title, updated_at = NOW()
                WHERE id = :id
            """),
            {"id": conversation_id, "title": title}
        )
        conn.commit()
        return result.rowcount > 0


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation."""
    with db_engine.connect() as conn:
        result = conn.execute(
            text("DELETE FROM conversations WHERE id = :id"),
            {"id": conversation_id}
        )
        conn.commit()
        return result.rowcount > 0


@app.get("/")
async def root():
    return {"status": "ok", "agent": "Dash - AI Data Analyst", "model": "Claude Opus 4"}


def run_stream_sync(message: str, session_id: str = None, language: str = "python"):
    """Synchronous generator that yields events from Dash agent.

    Uses a per-session lock to prevent concurrent requests from corrupting
    agent state (messages list, interpreter globals, stdout).
    """
    session = get_session(session_id)
    agent = session["agent"]
    lock = session["lock"]

    if not lock.acquire(timeout=60):
        yield {"type": "error", "error": "Session is busy with another request. Please wait."}
        return

    try:
        # Prepend language instruction to first message of session
        if language == "r":
            message = f"[USER PREFERS R] {message}"

        assistant_text = ""
        tool_results = []

        for event in agent.run(message, stream=True, stream_events=True):
            debug_log(f"Event received: {event.event} | {type(event).__name__}")

            if isinstance(event, ToolCallStarted):
                debug_log(f"Tool started: {event.tool_name}")
                yield {
                    "type": "tool_start",
                    "name": event.tool_name,
                    "args": event.tool_args,
                }

            elif isinstance(event, ToolCallCompleted):
                debug_log(f"Tool completed: {event.tool_name}")
                result = event.result

                # DON'T truncate if it contains a chart - we need the full base64/D3 data
                if '[CHART_BASE64]' in result:
                    debug_log(f"Chart detected, keeping full result ({len(result)} chars)")
                elif '[D3_CHART]' in result:
                    debug_log(f"D3 chart detected, keeping full result ({len(result)} chars)")
                elif len(result) > 2000:
                    result = result[:2000] + "... (truncated)"

                tool_results.append(result)
                yield {
                    "type": "tool_complete",
                    "name": event.tool_name,
                    "args": event.tool_args,
                    "result": result,
                }

            elif isinstance(event, TextDelta):
                if event.content:
                    assistant_text += event.content
                    yield {"type": "delta", "content": event.content}

            elif isinstance(event, StreamDone):
                yield {"type": "done"}

                # Generate follow-up suggestions in the background
                context_parts = [f"User: {message}"]
                for tr in tool_results[-3:]:
                    # Skip chart base64 data — just note a chart was made
                    if '[CHART_BASE64]' in tr:
                        context_parts.append("Tool: [chart generated]")
                    elif '[D3_CHART]' in tr:
                        context_parts.append("Tool: [interactive chart generated]")
                    else:
                        context_parts.append(f"Tool: {tr[:500]}")
                context_parts.append(f"Assistant: {assistant_text[:1000]}")
                hints = _generate_suggestions("\n".join(context_parts))
                if hints:
                    yield {"type": "suggestions", "suggestions": hints}

    except Exception as e:
        debug_log(f"Error: {e}")
        import traceback
        traceback.print_exc()
        yield {"type": "error", "error": str(e)}
    finally:
        lock.release()


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream the response with tool calls visible"""

    async def generate():
        loop = asyncio.get_event_loop()
        import queue
        import threading

        event_queue = queue.Queue()

        def producer():
            try:
                for event in run_stream_sync(request.message, request.session_id, request.language or "python"):
                    event_queue.put(event)
            finally:
                event_queue.put(None)  # Signal done

        thread = threading.Thread(target=producer)
        thread.start()

        while True:
            try:
                event = await loop.run_in_executor(None, lambda: event_queue.get(timeout=0.1))
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                continue

        thread.join()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming endpoint that returns full response"""
    try:
        tool_calls = []
        final_content = ""

        for event in run_stream_sync(request.message, request.session_id, request.language or "python"):
            if event["type"] == "tool_complete":
                tool_calls.append({
                    "name": event["name"],
                    "args": event["args"],
                    "result": event["result"],
                })
            elif event["type"] == "delta":
                final_content += event["content"]

        # Extract any charts from the response
        charts = []
        chart_pattern = r'\[CHART_BASE64\](.*?)\[/CHART_BASE64\]'
        for tool in tool_calls:
            if tool["result"]:
                for match in re.finditer(chart_pattern, str(tool["result"]), re.DOTALL):
                    charts.append(match.group(1))

        return {
            "response": final_content,
            "tool_calls": tool_calls,
            "charts": charts,
            "session_id": request.session_id or "default"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_session(request: ChatRequest):
    """Clear conversation history for a session"""
    session_id = request.session_id or "default"
    if session_id in sessions:
        session = sessions[session_id]
        with session["lock"]:
            session["agent"].clear_history()
            session["last_access"] = time.time()
        return {"status": "cleared", "session_id": session_id}
    return {"status": "no_session", "session_id": session_id}


@app.post("/restore")
async def restore_session(request: RestoreRequest):
    """Restore agent conversation history from saved frontend messages."""
    session_id = request.session_id or "default"
    session = get_session(session_id)
    with session["lock"]:
        session["agent"].restore_history(request.messages)
    return {"status": "restored", "session_id": session_id, "message_count": len(session["agent"].messages)}


# ============================================================
# Chart Fix (Vision Sub-Agent)
# ============================================================

def _extract_code_from_response(text: str, is_r: bool) -> str | None:
    """Extract a fenced code block from the vision model's response."""
    lang = 'r' if is_r else 'python'
    pattern = rf'```(?:{lang})?\s*\n([\s\S]*?)```'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: if entire response looks like code (no markdown fences)
    stripped = text.strip()
    if 'plt.' in stripped or 'ggplot' in stripped or 'fig' in stripped:
        return stripped
    return None


@app.post("/chart/fix")
async def fix_chart(request: ChartFixRequest):
    """Use vision sub-agent to fix visual issues in a chart, then re-execute.

    If execution fails (e.g. undefined variables), retries by feeding the error
    back to the vision model and asking it to make the code self-contained.
    """
    session = get_session(request.session_id)
    agent = session["agent"]
    lock = session["lock"]

    is_r = request.tool_name == "run_r_chart"
    lang_label = "R (ggplot2)" if is_r else "Python (matplotlib)"
    lang_fence = "r" if is_r else "python"

    MAX_ATTEMPTS = 3
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": request.chart_base64,
                },
            },
            {
                "type": "text",
                "text": f"Here is the {lang_label} code that generated this chart:\n\n```{lang_fence}\n{request.code}\n```\n\nFix overlapping labels and layout issues. Keep ALL variable names and data logic EXACTLY the same — only change layout/formatting parameters. Return the corrected code.",
            },
        ],
    }]

    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        # 1. Call vision model
        try:
            fix_response = _title_client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                system=CHART_FIX_SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Vision model call failed: {e}")

        response_text = fix_response.content[0].text
        corrected_code = _extract_code_from_response(response_text, is_r)
        if not corrected_code:
            raise HTTPException(status_code=422, detail="Could not extract corrected code from vision model response")

        # 2. Execute through the session's interpreter
        if not lock.acquire(timeout=30):
            raise HTTPException(status_code=409, detail="Session is busy with another request")
        try:
            if is_r:
                result = agent.r_interpreter.run_code_and_get_chart(corrected_code)
            else:
                result = agent.interpreter.run_code_and_get_chart(corrected_code)
        finally:
            lock.release()

        # 3. Check if chart was generated
        chart_match = re.search(r'\[CHART_BASE64\](.*?)\[/CHART_BASE64\]', result, re.DOTALL)
        if chart_match:
            return {
                "code": corrected_code,
                "chart_base64": chart_match.group(1),
            }

        # Execution failed — feed the error back for a retry
        last_error = result[:500]
        logger.warning(f"Chart fix attempt {attempt + 1}/{MAX_ATTEMPTS} failed: {last_error}")

        # Append assistant response + user error feedback for the retry conversation
        messages.append({"role": "assistant", "content": response_text})
        messages.append({
            "role": "user",
            "content": (
                f"That code failed with this error:\n\n```\n{last_error}\n```\n\n"
                "The variables referenced are not available. "
                "Rewrite the code to be COMPLETELY SELF-CONTAINED — "
                "define all data inline (use the original code's data or recreate it). "
                "Do NOT reference any external variables. Return only the corrected code."
            ),
        })

    raise HTTPException(status_code=422, detail=f"Chart fix failed after {MAX_ATTEMPTS} attempts: {last_error}")


# ============================================================
# Conversation Endpoints
# ============================================================

@app.post("/conversations")
async def api_create_conversation(request: ConversationCreate):
    """Create a new conversation"""
    try:
        conversation_id = create_conversation(request.title, request.language or "python")
        return {"id": conversation_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations")
async def api_list_conversations(limit: int = 50):
    """List all conversations"""
    try:
        conversations = list_conversations(limit)
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}")
async def api_get_conversation(conversation_id: str):
    """Get a specific conversation"""
    try:
        conversation = get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/conversations/{conversation_id}")
async def api_update_conversation(conversation_id: str, request: ConversationUpdate):
    """Update a conversation"""
    try:
        if request.title is not None:
            update_conversation_title(conversation_id, request.title)
        if request.messages is not None:
            update_conversation_messages(conversation_id, request.messages)
        return {"status": "updated", "id": conversation_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{conversation_id}")
async def api_delete_conversation(conversation_id: str):
    """Delete a conversation"""
    try:
        deleted = delete_conversation(conversation_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"status": "deleted", "id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversations/{conversation_id}/messages")
async def api_save_messages(conversation_id: str, request: ConversationUpdate):
    """Save messages to a conversation (called by frontend after each exchange)"""
    try:
        if request.messages is None:
            raise HTTPException(status_code=400, detail="Messages required")

        # Verify conversation exists
        conversation = get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Update messages
        update_conversation_messages(conversation_id, request.messages)

        # Auto-generate title with Haiku after first user message
        if not conversation.get("title") and request.messages:
            first_user_msg = next((m.get("content", "") for m in request.messages if m.get("role") == "user"), None)
            if first_user_msg:
                # Fire and forget — don't block the response
                def gen_title(conv_id, msg):
                    try:
                        resp = _title_client.messages.create(
                            model="claude-haiku-4-5-20251001",
                            max_tokens=20,
                            messages=[{"role": "user", "content": f"Title this chat: {msg}"}],
                            system="You are a title generator. Output 3-6 words, plain text only. No markdown, no bold, no asterisks, no quotes, no colons, no punctuation. Just the words. Examples: Stock Price Analysis for AMZN, F1 Championship Winners by Decade, TidyTuesday Water Access Data",
                        )
                        title = resp.content[0].text.strip().strip('"\'').replace('*', '').replace('#', '').replace(':', '')
                        update_conversation_title(conv_id, title)
                        debug_log(f"Generated title for {conv_id}: {title}")
                    except Exception as e:
                        debug_log(f"Title generation failed: {e}")
                        # Fallback to truncation
                        update_conversation_title(conv_id, msg[:50] + ("..." if len(msg) > 50 else ""))

                threading.Thread(target=gen_title, args=(conversation_id, first_user_msg), daemon=True).start()

        return {"status": "saved", "id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Kaggle dataset proxy
# ---------------------------------------------------------------------------
_kaggle_cache: dict[str, dict] = {}
KAGGLE_CACHE_TTL = 300  # 5 minutes


def _fmt_bytes(b: int) -> str:
    if not b:
        return "?"
    if b < 1024:
        return f"{b} B"
    if b < 1048576:
        return f"{b / 1024:.1f} KB"
    if b < 1073741824:
        return f"{b / 1048576:.1f} MB"
    return f"{b / 1073741824:.1f} GB"


@app.post("/kaggle/load")
async def kaggle_load_dataset(req: KaggleLoadRequest):
    """Download a Kaggle dataset and pre-populate a conversation.

    Runs the download + exploration code directly in the CodeInterpreter
    (no Claude API call), so the user sees results instantly.
    """
    import asyncio

    slug = req.ref.split("/")[1]
    download_url = f"https://www.kaggle.com/api/v1/datasets/download/{req.ref}"

    # 1. Fetch dataset description (data dictionary)
    description = ""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"https://www.kaggle.com/api/v1/datasets/view/{req.ref}")
            if resp.status_code == 200:
                description = resp.json().get("description", "")
    except Exception:
        pass  # proceed without description

    # 2. Create conversation in DB
    conversation_id = create_conversation(None, req.language or "python")

    # 3. Create session with a real DashAgent (so CodeInterpreter has state)
    session = get_session(conversation_id)
    agent = session["agent"]
    interp = agent.interpreter

    # 4. Run download + exploration in CodeInterpreter (sync — use executor)
    download_code = f'''import subprocess, os, glob
download_dir = os.path.expanduser("~/Downloads/kaggle-{slug}")
os.makedirs(download_dir, exist_ok=True)
subprocess.run(["curl", "-L", "-o", f"{{download_dir}}/{slug}.zip", "{download_url}"], check=True, capture_output=True)
subprocess.run(["unzip", "-o", f"{{download_dir}}/{slug}.zip", "-d", download_dir], check=True, capture_output=True)
csv_files = glob.glob(f"{{download_dir}}/*.csv")
print(f"Found {{len(csv_files)}} CSV files:", [os.path.basename(f) for f in csv_files])
'''

    explore_code = f'''import pandas as pd, os, glob
download_dir = os.path.expanduser("~/Downloads/kaggle-{slug}")
csv_files = sorted(glob.glob(f"{{download_dir}}/*.csv"))
summaries = []
for fp in csv_files[:5]:
    name = os.path.basename(fp)
    df = pd.read_csv(fp)
    globals()[name.replace(".csv", "").replace("-", "_").replace(" ", "_")] = df
    summaries.append(f"**{{name}}** — {{len(df):,}} rows × {{len(df.columns)}} columns")
    summaries.append(f"Columns: {{", ".join(df.columns.tolist())}}")
    summaries.append(f"Dtypes: {{dict(df.dtypes.value_counts())}}")
    summaries.append(df.head(5).to_string())
    summaries.append("")
print("\\n".join(summaries))
'''

    def run_sync():
        r1 = interp.run_code(download_code)
        r2 = interp.run_code(explore_code)
        return r1, r2

    loop = asyncio.get_event_loop()
    download_result, explore_result = await loop.run_in_executor(executor, run_sync)

    # 5. Build messages — clean user msg, description in collapsible context box
    user_content = f"Explore the {req.title} dataset from Kaggle"

    size_str = _fmt_bytes(req.totalBytes)
    assistant_text = f"I've downloaded and loaded **{req.title}** ({size_str}). Here's what's in the dataset:"

    events = [
        {"type": "tool", "tool": {"name": "run_code", "args": {"code": download_code.strip()}, "result": download_result}},
        {"type": "tool", "tool": {"name": "run_code", "args": {"code": explore_code.strip()}, "result": explore_result}},
    ]
    if description:
        events.append({"type": "context", "title": "Data dictionary", "content": description})
    events.append({"type": "text", "content": assistant_text + "\n\nWhat would you like to explore?"})

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "events": events},
    ]

    # 7. Seed agent message history (so Claude knows what happened)
    agent.restore_history(messages)

    # 8. Save messages to DB + generate title in background
    update_conversation_messages(conversation_id, messages)

    def gen_title(conv_id, title):
        try:
            resp = _title_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20,
                messages=[{"role": "user", "content": f"Title this chat: Exploring {title} dataset from Kaggle"}],
                system="You are a title generator. Output 3-6 words, plain text only. No markdown, no bold, no asterisks, no quotes, no colons, no punctuation. Just the words. Examples: Stock Price Analysis for AMZN, F1 Championship Winners by Decade, TidyTuesday Water Access Data",
            )
            t = resp.content[0].text.strip().strip('"\'').replace('*', '').replace('#', '').replace(':', '')
            update_conversation_title(conv_id, t)
        except Exception:
            update_conversation_title(conv_id, f"{title[:45]} Analysis")

    threading.Thread(target=gen_title, args=(conversation_id, req.title), daemon=True).start()

    hints = _generate_suggestions(explore_result, req.title, req.subtitle or "")

    return {
        "conversation_id": conversation_id,
        "messages": messages,
        "suggestions": hints,
    }


@app.get("/kaggle/datasets")
async def kaggle_list_datasets(
    search: str | None = None,
    sort_by: str = "hottest",
    page: int = 1,
):
    """Proxy Kaggle dataset listing to avoid CORS issues."""
    cache_key = f"{search}|{sort_by}|{page}"
    now = time.time()

    if cache_key in _kaggle_cache:
        entry = _kaggle_cache[cache_key]
        if now - entry["ts"] < KAGGLE_CACHE_TTL:
            return entry["data"]

    params: dict = {
        "sortBy": sort_by,
        "page": page,
        "filetype": "csv",
        "maxSize": 52428800,  # 50 MB
    }
    if search:
        params["search"] = search

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://www.kaggle.com/api/v1/datasets/list",
                params=params,
            )
            resp.raise_for_status()
            raw = resp.json()

        datasets = []
        for d in raw:
            datasets.append({
                "ref": d.get("ref", ""),
                "title": d.get("title", ""),
                "subtitle": d.get("subtitle", ""),
                "creatorName": d.get("creatorName", ""),
                "totalBytes": d.get("totalBytes", 0),
                "downloadCount": d.get("downloadCount", 0),
                "voteCount": d.get("voteCount", 0),
                "viewCount": d.get("viewCount", 0),
                "lastUpdated": d.get("lastUpdated", ""),
                "usabilityRating": d.get("usabilityRating", 0),
                "tags": [t.get("name", "") for t in d.get("tags", [])],
                "url": d.get("url", ""),
            })

        result = {"datasets": datasets, "page": page}
        _kaggle_cache[cache_key] = {"data": result, "ts": now}

        # Evict stale entries
        stale = [k for k, v in _kaggle_cache.items() if now - v["ts"] > KAGGLE_CACHE_TTL]
        for k in stale:
            del _kaggle_cache[k]

        return result

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Kaggle API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Kaggle datasets: {e}")


@app.get("/kaggle/datasets/{owner}/{slug}")
async def kaggle_dataset_detail(owner: str, slug: str):
    """Fetch full description/data dictionary for a single Kaggle dataset."""
    ref = f"{owner}/{slug}"
    cache_key = f"detail|{ref}"
    now = time.time()

    if cache_key in _kaggle_cache:
        entry = _kaggle_cache[cache_key]
        if now - entry["ts"] < KAGGLE_CACHE_TTL:
            return entry["data"]

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"https://www.kaggle.com/api/v1/datasets/view/{ref}")
            resp.raise_for_status()
            raw = resp.json()

        result = {
            "ref": raw.get("ref", ""),
            "title": raw.get("title", ""),
            "description": raw.get("description", ""),
        }
        _kaggle_cache[cache_key] = {"data": result, "ts": now}
        return result

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Kaggle API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch dataset detail: {e}")


# ---------------------------------------------------------------------------
# TidyTuesday dataset proxy
# ---------------------------------------------------------------------------
_tt_cache: dict[str, dict] = {}
TT_CACHE_TTL = 600  # 10 minutes (repo data changes infrequently)
_TT_RAW = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main"


def _parse_tt_table(md: str) -> list[dict]:
    """Parse the markdown table from a TidyTuesday year readme.md."""
    datasets = []
    in_table = False
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 3:
            continue
        # Skip header / separator rows
        if cells[0].replace("-", "").replace(":", "").strip() == "" or cells[0].lower() == "week":
            in_table = True
            continue
        if not in_table:
            continue

        week = cells[0].strip()
        date = cells[1].strip()
        data_cell = cells[2].strip() if len(cells) > 2 else ""
        source_cell = cells[3].strip() if len(cells) > 3 else ""
        article_cell = cells[4].strip() if len(cells) > 4 else ""

        # Parse [title](url) from data_cell
        title_match = re.match(r'\[([^\]]+)\]\(([^)]+)\)', data_cell)
        title = title_match.group(1) if title_match else data_cell

        # Parse source link(s)
        source_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', source_cell)
        source_name = source_links[0][0] if source_links else source_cell

        # Parse article link(s)
        article_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', article_cell)
        article_name = article_links[0][0] if article_links else article_cell

        if not date or date == "NA":
            continue

        datasets.append({
            "week": int(week) if week.isdigit() else 0,
            "date": date,
            "title": title,
            "source": source_name,
            "sourceUrl": source_links[0][1] if source_links else "",
            "article": article_name,
            "articleUrl": article_links[0][1] if article_links else "",
        })

    # Reverse so newest is first
    datasets.reverse()
    return datasets


@app.get("/tidytuesday/datasets")
async def tidytuesday_list_datasets(year: int = 2026):
    """List TidyTuesday datasets for a given year."""
    cache_key = f"tt|{year}"
    now = time.time()

    if cache_key in _tt_cache:
        entry = _tt_cache[cache_key]
        if now - entry["ts"] < TT_CACHE_TTL:
            return entry["data"]

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{_TT_RAW}/data/{year}/readme.md")
            resp.raise_for_status()
            md = resp.text

        datasets = _parse_tt_table(md)
        result = {"datasets": datasets, "year": year}
        _tt_cache[cache_key] = {"data": result, "ts": now}

        # Evict stale
        stale = [k for k, v in _tt_cache.items() if now - v["ts"] > TT_CACHE_TTL]
        for k in stale:
            del _tt_cache[k]

        return result

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"datasets": [], "year": year}
        raise HTTPException(status_code=e.response.status_code, detail=f"GitHub error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch TidyTuesday datasets: {e}")


@app.post("/tidytuesday/load")
async def tidytuesday_load_dataset(req: TidyTuesdayLoadRequest):
    """Download a TidyTuesday dataset and pre-populate a conversation."""
    import asyncio
    import yaml

    date = req.date
    year = req.year
    base_url = f"{_TT_RAW}/data/{year}/{date}"

    # 1. Fetch dataset description from readme.md
    description = ""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{base_url}/readme.md")
            if resp.status_code == 200:
                description = resp.text
    except Exception:
        pass

    # 2. List CSV/TSV files via GitHub API
    csv_files = []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"https://api.github.com/repos/rfordatascience/tidytuesday/contents/data/{year}/{date}",
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            if resp.status_code == 200:
                for f in resp.json():
                    name = f.get("name", "")
                    if name.endswith(".csv") or name.endswith(".tsv"):
                        csv_files.append(name)
    except Exception:
        pass

    if not csv_files:
        raise HTTPException(status_code=404, detail="No CSV files found for this dataset")

    # 3. Create conversation
    conversation_id = create_conversation(None, req.language or "python")
    session = get_session(conversation_id)
    agent = session["agent"]
    interp = agent.interpreter

    # 4. Build download code
    file_urls = {f: f"{base_url}/{f}" for f in csv_files}
    download_lines = [
        'import subprocess, os',
        f'download_dir = os.path.expanduser("~/Downloads/tidytuesday-{date}")',
        'os.makedirs(download_dir, exist_ok=True)',
    ]
    for fname, url in file_urls.items():
        download_lines.append(
            f'subprocess.run(["curl", "-sL", "-o", f"{{download_dir}}/{fname}", "{url}"], check=True, capture_output=True)'
        )
    download_lines.append(f'print("Downloaded {len(csv_files)} file(s):", {[f for f in csv_files]})')
    download_code = "\n".join(download_lines)

    explore_code = f'''import pandas as pd, os, glob
download_dir = os.path.expanduser("~/Downloads/tidytuesday-{date}")
csv_files = sorted([f for f in os.listdir(download_dir) if f.endswith(".csv") or f.endswith(".tsv")])
summaries = []
for fname in csv_files[:5]:
    fp = os.path.join(download_dir, fname)
    sep = "\\t" if fname.endswith(".tsv") else ","
    df = pd.read_csv(fp, sep=sep)
    var_name = fname.rsplit(".", 1)[0].replace("-", "_").replace(" ", "_")
    globals()[var_name] = df
    summaries.append(f"**{{fname}}** — {{len(df):,}} rows × {{len(df.columns)}} columns")
    summaries.append(f"Columns: {{", ".join(df.columns.tolist())}}")
    summaries.append(f"Dtypes: {{dict(df.dtypes.value_counts())}}")
    summaries.append(df.head(5).to_string())
    summaries.append("")
print("\\n".join(summaries))
'''

    def run_sync():
        r1 = interp.run_code(download_code)
        r2 = interp.run_code(explore_code)
        return r1, r2

    loop = asyncio.get_event_loop()
    download_result, explore_result = await loop.run_in_executor(executor, run_sync)

    # 5. Build messages — clean user msg, description in collapsible context box
    user_content = f"Explore the {req.title} TidyTuesday dataset"

    assistant_text = f"I've downloaded and loaded the **{req.title}** TidyTuesday dataset ({date}). Here's what's in it:"

    events = [
        {"type": "tool", "tool": {"name": "run_code", "args": {"code": download_code.strip()}, "result": download_result}},
        {"type": "tool", "tool": {"name": "run_code", "args": {"code": explore_code.strip()}, "result": explore_result}},
    ]
    if description:
        events.append({"type": "context", "title": "Dataset description", "content": description})
    events.append({"type": "text", "content": assistant_text + "\n\nWhat would you like to explore?"})

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "events": events},
    ]

    agent.restore_history(messages)
    update_conversation_messages(conversation_id, messages)

    # Title generation
    def gen_title(conv_id, title):
        try:
            resp = _title_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20,
                messages=[{"role": "user", "content": f"Title this chat: Exploring {title} dataset from TidyTuesday"}],
                system="You are a title generator. Output 3-6 words, plain text only. No markdown, no bold, no asterisks, no quotes, no colons, no punctuation. Just the words. Examples: Stock Price Analysis for AMZN, F1 Championship Winners by Decade, TidyTuesday Water Access Data",
            )
            t = resp.content[0].text.strip().strip('"\'').replace('*', '').replace('#', '').replace(':', '')
            update_conversation_title(conv_id, t)
        except Exception:
            update_conversation_title(conv_id, f"TidyTuesday {title[:40]}")

    threading.Thread(target=gen_title, args=(conversation_id, req.title), daemon=True).start()

    hints = _generate_suggestions(explore_result, req.title)

    return {
        "conversation_id": conversation_id,
        "messages": messages,
        "suggestions": hints,
    }


# ---------------------------------------------------------------------------
# Our World in Data
# ---------------------------------------------------------------------------

_OWID_CATALOG = [
    # Population & Demographics
    {"slug": "population", "title": "Population", "topic": "Population", "description": "Total population by country and region over centuries"},
    {"slug": "life-expectancy", "title": "Life expectancy", "topic": "Population", "description": "Life expectancy at birth by country"},
    {"slug": "total-fertility-rate", "title": "Fertility rate", "topic": "Population", "description": "Average number of children per woman"},
    {"slug": "child-mortality-igme", "title": "Child mortality", "topic": "Population", "description": "Under-five mortality rate per 1,000 live births"},
    {"slug": "median-age", "title": "Median age", "topic": "Population", "description": "Median age of the population by country"},
    {"slug": "urbanization-last-500-years", "title": "Urbanization", "topic": "Population", "description": "Share of population living in urban areas"},
    {"slug": "children-per-woman-un", "title": "Children per woman (UN)", "topic": "Population", "description": "Total fertility rate from UN population estimates"},

    # Health
    {"slug": "deaths-by-age-group", "title": "Deaths by age group", "topic": "Health", "description": "Number of deaths by age group over time"},
    {"slug": "share-of-adults-defined-as-obese", "title": "Obesity prevalence", "topic": "Health", "description": "Share of adults defined as obese (BMI >= 30)"},
    {"slug": "daily-per-capita-caloric-supply", "title": "Caloric supply per capita", "topic": "Health", "description": "Daily per capita caloric supply by country"},
    {"slug": "food-supply-kcal", "title": "Food supply", "topic": "Health", "description": "Food supply in kilocalories per person per day"},

    # Energy
    {"slug": "global-primary-energy", "title": "Global primary energy", "topic": "Energy", "description": "Primary energy consumption by source worldwide"},
    {"slug": "share-electricity-renewables", "title": "Renewable electricity share", "topic": "Energy", "description": "Share of electricity production from renewables"},
    {"slug": "renewable-share-energy", "title": "Renewable energy share", "topic": "Energy", "description": "Share of primary energy from renewable sources"},
    {"slug": "nuclear-energy-generation", "title": "Nuclear energy generation", "topic": "Energy", "description": "Electricity generation from nuclear power"},
    {"slug": "solar-energy-consumption", "title": "Solar energy consumption", "topic": "Energy", "description": "Primary energy consumption from solar"},
    {"slug": "share-electricity-coal", "title": "Electricity from coal", "topic": "Energy", "description": "Share of electricity production from coal"},
    {"slug": "share-electricity-nuclear", "title": "Electricity from nuclear", "topic": "Energy", "description": "Share of electricity production from nuclear"},
    {"slug": "share-electricity-gas", "title": "Electricity from gas", "topic": "Energy", "description": "Share of electricity production from natural gas"},
    {"slug": "share-electricity-solar", "title": "Electricity from solar", "topic": "Energy", "description": "Share of electricity production from solar"},
    {"slug": "share-electricity-wind", "title": "Electricity from wind", "topic": "Energy", "description": "Share of electricity production from wind"},

    # CO2 & Climate
    {"slug": "annual-co2-emissions", "title": "Annual CO2 emissions", "topic": "CO2 & Climate", "description": "Annual carbon dioxide emissions by country"},
    {"slug": "co2-emissions-by-fuel-line", "title": "CO2 emissions by fuel", "topic": "CO2 & Climate", "description": "CO2 emissions by fuel type (coal, oil, gas, cement)"},
    {"slug": "cumulative-co2-emissions-region", "title": "Cumulative CO2 emissions", "topic": "CO2 & Climate", "description": "Cumulative CO2 emissions by world region"},
    {"slug": "temperature-anomaly", "title": "Temperature anomaly", "topic": "CO2 & Climate", "description": "Global average temperature anomaly relative to 1961-1990"},
    {"slug": "sea-level-rise", "title": "Sea level rise", "topic": "CO2 & Climate", "description": "Global mean sea level change"},

    # Poverty & Economy
    {"slug": "share-of-population-in-extreme-poverty", "title": "Extreme poverty", "topic": "Poverty & Economy", "description": "Share of population living on less than $2.15/day"},
    {"slug": "gdp-per-capita-worldbank", "title": "GDP per capita", "topic": "Poverty & Economy", "description": "GDP per capita adjusted for inflation (constant USD)"},
    {"slug": "human-development-index", "title": "Human Development Index", "topic": "Poverty & Economy", "description": "Composite index of life expectancy, education, and income"},
    {"slug": "economic-inequality-gini-index", "title": "Gini index", "topic": "Poverty & Economy", "description": "Income inequality measured by the Gini coefficient"},
    {"slug": "income-share-held-by-richest-10", "title": "Income share of richest 10%", "topic": "Poverty & Economy", "description": "Share of pre-tax income held by the richest 10%"},

    # Food & Agriculture
    {"slug": "global-meat-production", "title": "Global meat production", "topic": "Food & Agriculture", "description": "Meat production by livestock type worldwide"},
    {"slug": "share-of-land-area-used-for-agriculture", "title": "Agricultural land use", "topic": "Food & Agriculture", "description": "Share of land area used for agriculture"},
    {"slug": "cereal-crop-yield-vs-fertilizer-application", "title": "Crop yield vs fertilizer", "topic": "Food & Agriculture", "description": "Cereal yield compared to fertilizer use by country"},

    # Technology
    {"slug": "number-of-internet-users", "title": "Internet users", "topic": "Technology", "description": "Number of people using the internet by country"},
    {"slug": "share-of-individuals-using-the-internet", "title": "Internet penetration", "topic": "Technology", "description": "Share of population using the internet"},
    {"slug": "mobile-cellular-subscriptions-per-100-people", "title": "Mobile subscriptions", "topic": "Technology", "description": "Mobile cellular subscriptions per 100 people"},
    {"slug": "technology-adoption-by-households-in-the-united-states", "title": "Technology adoption (US)", "topic": "Technology", "description": "Adoption rates of technologies in American households"},

    # Education
    {"slug": "mean-years-of-schooling", "title": "Mean years of schooling", "topic": "Education", "description": "Average years of schooling for adults 25+"},
    {"slug": "literacy-rate-by-country", "title": "Literacy rate", "topic": "Education", "description": "Share of adults who can read and write"},
    {"slug": "total-government-expenditure-on-education-gdp", "title": "Education spending (% GDP)", "topic": "Education", "description": "Government expenditure on education as share of GDP"},
    {"slug": "government-expenditure-education", "title": "Education spending", "topic": "Education", "description": "Total government expenditure on education"},

    # Democracy & Governance
    {"slug": "electoral-democracy-index", "title": "Electoral democracy index", "topic": "Democracy", "description": "V-Dem electoral democracy index (0-1 scale)"},
    {"slug": "political-regime", "title": "Political regime", "topic": "Democracy", "description": "Classification of political regimes by country"},
]

_OWID_TOPICS = sorted(set(d["topic"] for d in _OWID_CATALOG))


@app.get("/owid/datasets")
async def owid_list_datasets(topic: str | None = None):
    """List curated Our World in Data datasets, optionally filtered by topic."""
    datasets = _OWID_CATALOG
    if topic and topic != "All":
        datasets = [d for d in datasets if d["topic"] == topic]
    return {"datasets": datasets, "topics": _OWID_TOPICS}


@app.post("/owid/load")
async def owid_load_dataset(req: OwidLoadRequest):
    """Download an OWID dataset and pre-populate a conversation."""
    import asyncio

    slug = req.slug
    csv_url = f"https://ourworldindata.org/grapher/{slug}.csv"
    meta_url = f"https://ourworldindata.org/grapher/{slug}.metadata.json"

    # 1. Fetch metadata (title, subtitle, sources)
    description = ""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(meta_url)
            if resp.status_code == 200:
                meta = resp.json()
                chart = meta.get("chart", {})
                parts = []
                if chart.get("title"):
                    parts.append(chart["title"])
                if chart.get("subtitle"):
                    parts.append(chart["subtitle"])
                # Add column descriptions
                for col_name, col_info in meta.get("columns", {}).items():
                    if col_name not in ("Entity", "Code", "Year", "Day"):
                        unit = col_info.get("unit", "")
                        desc = col_info.get("description", "")
                        if desc:
                            parts.append(f"**{col_name}**: {desc}" + (f" ({unit})" if unit else ""))
                description = "\n\n".join(parts)
    except Exception:
        pass

    # 2. Download CSV with httpx (urllib User-Agent gets 403'd by Cloudflare)
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        csv_resp = await client.get(csv_url)
        csv_resp.raise_for_status()
    csv_bytes = csv_resp.content

    # 3. Create conversation
    conversation_id = create_conversation(None, req.language or "python")
    session = get_session(conversation_id)
    agent = session["agent"]
    interp = agent.interpreter

    # Write CSV to interpreter's temp dir so CodeInterpreter loads locally
    import os
    csv_path = os.path.join(interp._temp_dir, f"{slug}.csv")
    with open(csv_path, "wb") as f:
        f.write(csv_bytes)

    # 4. Explore the local CSV (display_code shows the URL for the user)
    var_name = slug.replace("-", "_")
    explore_tail = f'''summaries = []
summaries.append(f"**{slug}.csv** — {{{var_name}.shape[0]:,}} rows × {{{var_name}.shape[1]}} columns")
summaries.append("Columns: " + ", ".join({var_name}.columns.tolist()))
summaries.append(f"Dtypes: {{dict({var_name}.dtypes.value_counts())}}")
summaries.append({var_name}.head(5).to_string())
print("\\n".join(summaries))'''
    display_code = f'import pandas as pd\n{var_name} = pd.read_csv("{csv_url}")\n{explore_tail}'
    load_code = f'import pandas as pd\n{var_name} = pd.read_csv("{slug}.csv")\n{explore_tail}\n'

    def run_sync():
        return interp.run_code(load_code)

    loop = asyncio.get_event_loop()
    explore_result = await loop.run_in_executor(executor, run_sync)

    # 4. Build messages
    user_content = f"I want to explore Our World in Data: **{req.title}**"
    assistant_text = f"I've loaded **{req.title}** from Our World in Data. Here's what's in the dataset:"

    events = [
        {"type": "tool", "tool": {"name": "run_code", "args": {"code": display_code.strip()}, "result": explore_result}},
    ]
    if description:
        events.append({"type": "context", "title": "Dataset description", "content": description})
    events.append({"type": "text", "content": assistant_text + "\n\nWhat would you like to explore?"})

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "events": events},
    ]

    # 5. Seed agent + save + title
    agent.restore_history(messages)
    update_conversation_messages(conversation_id, messages)

    def gen_title(conv_id, title):
        try:
            resp = _title_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20,
                messages=[{"role": "user", "content": f"Title this chat: Exploring {title} from Our World in Data"}],
                system="You are a title generator. Output 3-6 words, plain text only. No markdown, no bold, no asterisks, no quotes, no colons, no punctuation. Just the words. Examples: Stock Price Analysis for AMZN, F1 Championship Winners by Decade, TidyTuesday Water Access Data",
            )
            t = resp.content[0].text.strip().strip('"\'').replace('*', '').replace('#', '').replace(':', '')
            update_conversation_title(conv_id, t)
        except Exception:
            update_conversation_title(conv_id, f"OWID {title[:40]}")

    threading.Thread(target=gen_title, args=(conversation_id, req.title), daemon=True).start()

    hints = _generate_suggestions(explore_result, req.title, description)

    return {
        "conversation_id": conversation_id,
        "messages": messages,
        "suggestions": hints,
    }


# ---------------------------------------------------------------------------
# FRED (Federal Reserve Economic Data)
# ---------------------------------------------------------------------------

_FRED_CATALOG = [
    # GDP & Growth
    {"series_id": "GDP", "title": "Gross Domestic Product", "category": "GDP & Growth", "description": "Nominal GDP, quarterly, seasonally adjusted annual rate", "frequency": "Quarterly"},
    {"series_id": "GDPC1", "title": "Real Gross Domestic Product", "category": "GDP & Growth", "description": "Real GDP in chained 2017 dollars, quarterly", "frequency": "Quarterly"},
    {"series_id": "A191RL1Q225SBEA", "title": "Real GDP Growth Rate", "category": "GDP & Growth", "description": "Percent change from preceding period, quarterly", "frequency": "Quarterly"},
    {"series_id": "GDPPOT", "title": "Real Potential GDP", "category": "GDP & Growth", "description": "Congressional Budget Office estimate of potential output", "frequency": "Quarterly"},
    # Employment
    {"series_id": "UNRATE", "title": "Unemployment Rate", "category": "Employment", "description": "Civilian unemployment rate, seasonally adjusted", "frequency": "Monthly"},
    {"series_id": "PAYEMS", "title": "Total Nonfarm Payrolls", "category": "Employment", "description": "All employees, total nonfarm, seasonally adjusted", "frequency": "Monthly"},
    {"series_id": "ICSA", "title": "Initial Jobless Claims", "category": "Employment", "description": "Initial claims for unemployment insurance, weekly", "frequency": "Weekly"},
    {"series_id": "CIVPART", "title": "Labor Force Participation Rate", "category": "Employment", "description": "Civilian labor force participation rate", "frequency": "Monthly"},
    {"series_id": "U6RATE", "title": "U-6 Unemployment Rate", "category": "Employment", "description": "Total unemployed plus marginally attached plus part-time for economic reasons", "frequency": "Monthly"},
    # Inflation
    {"series_id": "CPIAUCSL", "title": "Consumer Price Index (CPI)", "category": "Inflation", "description": "CPI for all urban consumers, all items, seasonally adjusted", "frequency": "Monthly"},
    {"series_id": "CPILFESL", "title": "Core CPI (ex Food & Energy)", "category": "Inflation", "description": "CPI less food and energy, seasonally adjusted", "frequency": "Monthly"},
    {"series_id": "PCEPI", "title": "PCE Price Index", "category": "Inflation", "description": "Personal consumption expenditures price index", "frequency": "Monthly"},
    {"series_id": "PCEPILFE", "title": "Core PCE Price Index", "category": "Inflation", "description": "PCE excluding food and energy — the Fed's preferred inflation gauge", "frequency": "Monthly"},
    # Interest Rates
    {"series_id": "FEDFUNDS", "title": "Federal Funds Rate", "category": "Interest Rates", "description": "Effective federal funds rate", "frequency": "Monthly"},
    {"series_id": "DGS10", "title": "10-Year Treasury Yield", "category": "Interest Rates", "description": "Market yield on 10-year Treasury constant maturity", "frequency": "Daily"},
    {"series_id": "DGS2", "title": "2-Year Treasury Yield", "category": "Interest Rates", "description": "Market yield on 2-year Treasury constant maturity", "frequency": "Daily"},
    {"series_id": "T10Y2Y", "title": "10Y-2Y Treasury Spread", "category": "Interest Rates", "description": "Spread between 10-year and 2-year Treasury — yield curve indicator", "frequency": "Daily"},
    {"series_id": "MORTGAGE30US", "title": "30-Year Mortgage Rate", "category": "Interest Rates", "description": "30-year fixed rate mortgage average", "frequency": "Weekly"},
    # Housing
    {"series_id": "MSPUS", "title": "Median Home Sale Price", "category": "Housing", "description": "Median sales price of houses sold in the US", "frequency": "Quarterly"},
    {"series_id": "HOUST", "title": "Housing Starts", "category": "Housing", "description": "New privately-owned housing units started", "frequency": "Monthly"},
    {"series_id": "CSUSHPISA", "title": "Case-Shiller Home Price Index", "category": "Housing", "description": "S&P/Case-Shiller U.S. National Home Price Index", "frequency": "Monthly"},
    {"series_id": "PERMIT", "title": "Building Permits", "category": "Housing", "description": "New privately-owned housing units authorized by building permits", "frequency": "Monthly"},
    # Consumer
    {"series_id": "UMCSENT", "title": "Consumer Sentiment", "category": "Consumer", "description": "University of Michigan consumer sentiment index", "frequency": "Monthly"},
    {"series_id": "RSAFS", "title": "Retail Sales", "category": "Consumer", "description": "Advance retail sales: retail and food services, seasonally adjusted", "frequency": "Monthly"},
    {"series_id": "DSPIC96", "title": "Real Disposable Income", "category": "Consumer", "description": "Real disposable personal income, billions of chained 2017 dollars", "frequency": "Monthly"},
    {"series_id": "PCE", "title": "Personal Consumption Expenditures", "category": "Consumer", "description": "Total personal consumption expenditures", "frequency": "Monthly"},
    # Money & Markets
    {"series_id": "M2SL", "title": "M2 Money Supply", "category": "Money & Markets", "description": "M2 money stock, seasonally adjusted", "frequency": "Monthly"},
    {"series_id": "SP500", "title": "S&P 500", "category": "Money & Markets", "description": "S&P 500 index level", "frequency": "Daily"},
    {"series_id": "VIXCLS", "title": "VIX Volatility Index", "category": "Money & Markets", "description": "CBOE volatility index — market fear gauge", "frequency": "Daily"},
    {"series_id": "DEXUSEU", "title": "USD/EUR Exchange Rate", "category": "Money & Markets", "description": "US dollars to one euro", "frequency": "Daily"},
    # Trade
    {"series_id": "BOPGSTB", "title": "Trade Balance", "category": "Trade", "description": "Trade balance on goods and services, seasonally adjusted", "frequency": "Monthly"},
    {"series_id": "DTWEXBGS", "title": "Trade Weighted Dollar Index", "category": "Trade", "description": "Nominal broad US dollar index", "frequency": "Daily"},
]


@app.get("/fred/datasets")
async def fred_list_datasets(category: str | None = None):
    datasets = _FRED_CATALOG
    if category:
        datasets = [d for d in datasets if d["category"] == category]
    categories = sorted(set(d["category"] for d in _FRED_CATALOG))
    return {"datasets": datasets, "categories": categories}


@app.post("/fred/load")
async def fred_load_dataset(req: FredLoadRequest):
    """Download a FRED series and pre-populate a conversation."""
    import asyncio

    series_id = req.series_id
    csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    conversation_id = create_conversation(None, req.language or "python")
    session = get_session(conversation_id)
    agent = session["agent"]
    interp = agent.interpreter

    var_name = series_id.lower().replace("-", "_")
    load_code = f'''import pandas as pd
{var_name} = pd.read_csv("{csv_url}")
{var_name}.columns = ["date", "value"]
{var_name}["date"] = pd.to_datetime({var_name}["date"])
{var_name}["value"] = pd.to_numeric({var_name}["value"], errors="coerce")
{var_name} = {var_name}.dropna(subset=["value"])
summaries = []
summaries.append(f"**{series_id}** — {{{var_name}.shape[0]:,}} observations")
summaries.append(f"Date range: {{{var_name}['date'].min().strftime('%Y-%m-%d')}} to {{{var_name}['date'].max().strftime('%Y-%m-%d')}}")
summaries.append(f"Value range: {{{var_name}['value'].min():,.2f}} to {{{var_name}['value'].max():,.2f}}")
summaries.append(f"Latest: {{{var_name}['value'].iloc[-1]:,.2f}} ({{{var_name}['date'].iloc[-1].strftime('%Y-%m-%d')}})")
summaries.append("Columns: " + ", ".join({var_name}.columns.tolist()))
summaries.append({var_name}.tail(10).to_string(index=False))
print("\\n".join(summaries))
'''

    def run_sync():
        return interp.run_code(load_code)

    loop = asyncio.get_event_loop()
    explore_result = await loop.run_in_executor(executor, run_sync)

    # Find catalog description
    cat_entry = next((d for d in _FRED_CATALOG if d["series_id"] == series_id), None)
    description = ""
    if cat_entry:
        description = f"{cat_entry['title']}: {cat_entry['description']} ({cat_entry['frequency']})"

    user_content = f"I want to explore FRED series: **{req.title}** ({series_id})"
    assistant_text = f"I've loaded **{req.title}** ({series_id}) from FRED. Here's what's in the data:"

    events = [
        {"type": "tool", "tool": {"name": "run_code", "args": {"code": load_code.strip()}, "result": explore_result}},
    ]
    if description:
        events.append({"type": "context", "title": "Series info", "content": description})
    events.append({"type": "text", "content": assistant_text + "\n\nWhat would you like to explore?"})

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "events": events},
    ]

    agent.restore_history(messages)
    update_conversation_messages(conversation_id, messages)

    def gen_title(conv_id, title):
        try:
            resp = _title_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20,
                messages=[{"role": "user", "content": f"Title this chat: Analyzing {title} from FRED economic data"}],
                system="You are a title generator. Output 3-6 words, plain text only. No markdown, no bold, no asterisks, no quotes, no colons, no punctuation. Just the words.",
            )
            t = resp.content[0].text.strip().strip('"\'').replace('*', '').replace('#', '').replace(':', '')
            update_conversation_title(conv_id, t)
        except Exception:
            update_conversation_title(conv_id, f"FRED {series_id} Analysis")

    threading.Thread(target=gen_title, args=(conversation_id, req.title), daemon=True).start()

    hints = _generate_suggestions(explore_result, req.title, description)

    return {
        "conversation_id": conversation_id,
        "messages": messages,
        "suggestions": hints,
    }


# ---------------------------------------------------------------------------
# World Bank Open Data
# ---------------------------------------------------------------------------

_WB_CATALOG = [
    # Economy
    {"indicator_id": "NY.GDP.MKTP.CD", "title": "GDP (current US$)", "category": "Economy", "description": "Gross domestic product at current US dollars"},
    {"indicator_id": "NY.GDP.PCAP.CD", "title": "GDP per capita (current US$)", "category": "Economy", "description": "GDP divided by midyear population"},
    {"indicator_id": "NY.GDP.MKTP.KD.ZG", "title": "GDP Growth (annual %)", "category": "Economy", "description": "Annual percentage growth rate of GDP at market prices"},
    {"indicator_id": "FP.CPI.TOTL.ZG", "title": "Inflation, Consumer Prices (annual %)", "category": "Economy", "description": "Annual percentage change in consumer price index"},
    {"indicator_id": "SL.UEM.TOTL.ZS", "title": "Unemployment (% of labor force)", "category": "Economy", "description": "Share of labor force without work but available and seeking employment"},
    {"indicator_id": "GC.DOD.TOTL.GD.ZS", "title": "Central Government Debt (% of GDP)", "category": "Economy", "description": "Entire stock of direct government obligations"},
    # Population
    {"indicator_id": "SP.POP.TOTL", "title": "Population, total", "category": "Population", "description": "Total population based on the de facto definition"},
    {"indicator_id": "SP.POP.GROW", "title": "Population Growth (annual %)", "category": "Population", "description": "Annual population growth rate"},
    {"indicator_id": "SP.URB.TOTL.IN.ZS", "title": "Urban Population (% of total)", "category": "Population", "description": "People living in urban areas as share of total population"},
    {"indicator_id": "SP.DYN.LE00.IN", "title": "Life Expectancy at Birth", "category": "Population", "description": "Average number of years a newborn is expected to live"},
    {"indicator_id": "SP.DYN.TFRT.IN", "title": "Fertility Rate (births per woman)", "category": "Population", "description": "Total fertility rate — births per woman"},
    # Health
    {"indicator_id": "SH.XPD.CHEX.GD.ZS", "title": "Health Expenditure (% of GDP)", "category": "Health", "description": "Current health expenditure as percentage of GDP"},
    {"indicator_id": "SH.DYN.MORT", "title": "Under-5 Mortality Rate (per 1,000)", "category": "Health", "description": "Probability of dying between birth and age 5 per 1,000 live births"},
    {"indicator_id": "SP.DYN.IMRT.IN", "title": "Infant Mortality Rate (per 1,000)", "category": "Health", "description": "Number of infants dying before age 1 per 1,000 live births"},
    {"indicator_id": "SH.MED.BEDS.ZS", "title": "Hospital Beds (per 1,000 people)", "category": "Health", "description": "Inpatient beds available in public and private hospitals"},
    # Education
    {"indicator_id": "SE.ADT.LITR.ZS", "title": "Literacy Rate, Adult Total (%)", "category": "Education", "description": "Percentage of people aged 15+ who can read and write"},
    {"indicator_id": "SE.XPD.TOTL.GD.ZS", "title": "Education Spending (% of GDP)", "category": "Education", "description": "Government expenditure on education as percentage of GDP"},
    {"indicator_id": "SE.PRM.ENRR", "title": "School Enrollment, Primary (% gross)", "category": "Education", "description": "Gross enrollment ratio in primary education"},
    {"indicator_id": "SE.SEC.ENRR", "title": "School Enrollment, Secondary (% gross)", "category": "Education", "description": "Gross enrollment ratio in secondary education"},
    # Environment
    {"indicator_id": "EN.ATM.CO2E.PC", "title": "CO2 Emissions (metric tons per capita)", "category": "Environment", "description": "Carbon dioxide emissions per person from fossil fuel burning"},
    {"indicator_id": "EG.USE.ELEC.KH.PC", "title": "Electric Power Consumption (kWh per capita)", "category": "Environment", "description": "Per capita electric power consumption"},
    {"indicator_id": "AG.LND.FRST.ZS", "title": "Forest Area (% of land area)", "category": "Environment", "description": "Land under natural or planted trees of at least 5 meters"},
    {"indicator_id": "EG.FEC.RNEW.ZS", "title": "Renewable Energy (% of total)", "category": "Environment", "description": "Renewable energy share of total final energy consumption"},
    # Trade
    {"indicator_id": "NE.EXP.GNFS.ZS", "title": "Exports (% of GDP)", "category": "Trade", "description": "Value of goods and services sold abroad as percentage of GDP"},
    {"indicator_id": "NE.IMP.GNFS.ZS", "title": "Imports (% of GDP)", "category": "Trade", "description": "Value of goods and services purchased from abroad as percentage of GDP"},
    {"indicator_id": "BX.KLT.DINV.WD.GD.ZS", "title": "Foreign Direct Investment, Net Inflows (% of GDP)", "category": "Trade", "description": "Net inflows of investment to acquire a lasting management interest"},
    {"indicator_id": "TG.VAL.TOTL.GD.ZS", "title": "Merchandise Trade (% of GDP)", "category": "Trade", "description": "Sum of merchandise exports and imports divided by GDP"},
    # Poverty & Inequality
    {"indicator_id": "SI.POV.DDAY", "title": "Poverty Headcount at $2.15/day (%)", "category": "Poverty & Inequality", "description": "Percentage of population living on less than $2.15 a day"},
    {"indicator_id": "SI.POV.GINI", "title": "Gini Index", "category": "Poverty & Inequality", "description": "Measure of income inequality — 0 is perfect equality, 100 is perfect inequality"},
    {"indicator_id": "SI.DST.10TH.10", "title": "Income Share Held by Top 10%", "category": "Poverty & Inequality", "description": "Percentage of income held by highest 10% of population"},
    # Technology
    {"indicator_id": "IT.NET.USER.ZS", "title": "Internet Users (% of population)", "category": "Technology", "description": "Percentage of individuals using the Internet"},
    {"indicator_id": "IT.CEL.SETS.P2", "title": "Mobile Subscriptions (per 100 people)", "category": "Technology", "description": "Mobile cellular subscriptions per 100 people"},
]


@app.get("/worldbank/datasets")
async def worldbank_list_datasets(category: str | None = None):
    datasets = _WB_CATALOG
    if category:
        datasets = [d for d in datasets if d["category"] == category]
    categories = sorted(set(d["category"] for d in _WB_CATALOG))
    return {"datasets": datasets, "categories": categories}


@app.post("/worldbank/load")
async def worldbank_load_dataset(req: WorldBankLoadRequest):
    """Download a World Bank indicator and pre-populate a conversation."""
    import asyncio

    indicator_id = req.indicator_id
    api_url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator_id}?format=json&per_page=20000&date=1960:2025&source=2"

    conversation_id = create_conversation(None, req.language or "python")
    session = get_session(conversation_id)
    agent = session["agent"]
    interp = agent.interpreter

    var_name = indicator_id.lower().replace(".", "_")
    load_code = f'''import pandas as pd
import json
import urllib.request

url = "{api_url}"
with urllib.request.urlopen(url) as resp:
    raw = json.loads(resp.read())

records = []
for r in raw[1]:
    if r["value"] is not None:
        records.append({{
            "country": r["country"]["value"],
            "country_code": r["countryiso3code"],
            "year": int(r["date"]),
            "value": r["value"]
        }})

{var_name} = pd.DataFrame(records).sort_values(["country", "year"]).reset_index(drop=True)
summaries = []
summaries.append(f"**{indicator_id}** — {{{var_name}.shape[0]:,}} observations")
summaries.append(f"Countries: {{{var_name}['country'].nunique()}} | Years: {{{var_name}['year'].min()}}–{{{var_name}['year'].max()}}")
summaries.append("Columns: " + ", ".join({var_name}.columns.tolist()))
summaries.append(f"Top 5 countries (latest year):")
latest_year = {var_name}["year"].max()
top5 = {var_name}[{var_name}["year"] == latest_year].nlargest(5, "value")[["country", "year", "value"]]
summaries.append(top5.to_string(index=False))
print("\\n".join(summaries))
'''

    def run_sync():
        return interp.run_code(load_code)

    loop = asyncio.get_event_loop()
    explore_result = await loop.run_in_executor(executor, run_sync)

    cat_entry = next((d for d in _WB_CATALOG if d["indicator_id"] == indicator_id), None)
    description = cat_entry["description"] if cat_entry else ""

    user_content = f"I want to explore World Bank data: **{req.title}**"
    assistant_text = f"I've loaded **{req.title}** from the World Bank. Here's what's in the data:"

    events = [
        {"type": "tool", "tool": {"name": "run_code", "args": {"code": load_code.strip()}, "result": explore_result}},
    ]
    if description:
        events.append({"type": "context", "title": "Indicator info", "content": description})
    events.append({"type": "text", "content": assistant_text + "\n\nWhat would you like to explore?"})

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "events": events},
    ]

    agent.restore_history(messages)
    update_conversation_messages(conversation_id, messages)

    def gen_title(conv_id, title):
        try:
            resp = _title_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20,
                messages=[{"role": "user", "content": f"Title this chat: Analyzing {title} from World Bank data"}],
                system="You are a title generator. Output 3-6 words, plain text only. No markdown, no bold, no asterisks, no quotes, no colons, no punctuation. Just the words.",
            )
            t = resp.content[0].text.strip().strip('"\'').replace('*', '').replace('#', '').replace(':', '')
            update_conversation_title(conv_id, t)
        except Exception:
            update_conversation_title(conv_id, f"World Bank {title[:40]}")

    threading.Thread(target=gen_title, args=(conversation_id, req.title), daemon=True).start()

    hints = _generate_suggestions(explore_result, req.title, description)

    return {
        "conversation_id": conversation_id,
        "messages": messages,
        "suggestions": hints,
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("  Dash API Server (Pure Anthropic SDK)")
    print("=" * 50)
    print("\n API docs: http://localhost:8000/docs")
    print(" Chat endpoint: POST http://localhost:8000/chat")
    print(" Stream endpoint: POST http://localhost:8000/chat/stream")
    print(" Clear session: POST http://localhost:8000/clear\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
