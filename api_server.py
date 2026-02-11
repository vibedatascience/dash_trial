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

# Lightweight client for title generation
_title_client = anthropic.Anthropic()

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
                "agent": DashAgent(db_url=db_url),
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


class RestoreRequest(BaseModel):
    session_id: str | None = None
    messages: list = []


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

                yield {
                    "type": "tool_complete",
                    "name": event.tool_name,
                    "args": event.tool_args,
                    "result": result,
                }

            elif isinstance(event, TextDelta):
                if event.content:
                    yield {"type": "delta", "content": event.content}

            elif isinstance(event, StreamDone):
                yield {"type": "done"}

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
