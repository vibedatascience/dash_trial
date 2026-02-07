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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

# API keys should be set via environment variables
# ANTHROPIC_API_KEY - for Claude model

from dash.dash_agent import (
    DashAgent,
    ToolCallStarted,
    ToolCallCompleted,
    TextDelta,
    StreamDone
)
from db.url import db_url

app = FastAPI(title="Dash API")
executor = ThreadPoolExecutor(max_workers=4)

# Store agents by session_id for conversation persistence
agents: dict[str, DashAgent] = {}


def debug_log(msg):
    """Print debug message to stderr"""
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)


def get_agent(session_id: str | None) -> DashAgent:
    """Get or create an agent for the session."""
    if session_id is None:
        session_id = "default"

    if session_id not in agents:
        debug_log(f"Creating new agent for session: {session_id}")
        agents[session_id] = DashAgent(db_url=db_url)

    return agents[session_id]


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


@app.get("/")
async def root():
    return {"status": "ok", "agent": "Dash (Pure Anthropic)", "model": "Claude Opus 4"}


def run_stream_sync(message: str, session_id: str = None):
    """Synchronous generator that yields events from Dash agent.

    Converts our event objects to the same format the old Agno implementation used.
    """
    try:
        agent = get_agent(session_id)

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

                # DON'T truncate if it contains a chart - we need the full base64
                if '[CHART_BASE64]' in result:
                    debug_log(f"Chart detected, keeping full result ({len(result)} chars)")
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
                for event in run_stream_sync(request.message, request.session_id):
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

        for event in run_stream_sync(request.message, request.session_id):
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
    if session_id in agents:
        agents[session_id].clear_history()
        return {"status": "cleared", "session_id": session_id}
    return {"status": "no_session", "session_id": session_id}


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
