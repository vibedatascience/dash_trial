"""
Simple API server for Dash with streaming
Run: python api_server.py
"""

import os
import json
import asyncio
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

# API keys should be set via environment variables
# OPENAI_API_KEY - for embeddings
# ANTHROPIC_API_KEY - for Claude model

from dash.agents import dash

app = FastAPI(title="Dash API")
executor = ThreadPoolExecutor(max_workers=4)

def debug_log(msg):
    """Print debug message to stderr"""
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)

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
    return {"status": "ok", "agent": "Dash", "model": "Claude Opus 4.6"}


def run_stream_sync(message: str, session_id: str = None):
    """Synchronous generator that yields events from Dash"""
    try:
        # stream_events=True is CRITICAL for tool call events!
        # session_id is CRITICAL for chat history persistence!
        response_stream = dash.run(
            message,
            stream=True,
            stream_events=True,
            session_id=session_id,
        )

        for event in response_stream:
            event_type = getattr(event, 'event', None)
            debug_log(f"Event received: {event_type} | {type(event).__name__}")

            # Tool call started
            if event_type == 'ToolCallStarted':
                tool = getattr(event, 'tool', None)
                debug_log(f"Tool started: {tool}")
                if tool:
                    yield {
                        "type": "tool_start",
                        "name": getattr(tool, 'tool_name', 'unknown'),
                        "args": getattr(tool, 'tool_args', {}),
                    }

            # Tool call completed
            elif event_type == 'ToolCallCompleted':
                tool = getattr(event, 'tool', None)
                debug_log(f"Tool completed: {tool}")
                if tool:
                    result = getattr(tool, 'result', None)
                    result_str = str(result) if result else ""

                    # DON'T truncate if it contains a chart - we need the full base64
                    if '[CHART_BASE64]' in result_str:
                        # Keep full result for charts
                        debug_log(f"Chart detected, keeping full result ({len(result_str)} chars)")
                    elif len(result_str) > 2000:
                        # Truncate non-chart results
                        result = result_str[:2000] + "... (truncated)"

                    yield {
                        "type": "tool_complete",
                        "name": getattr(tool, 'tool_name', 'unknown'),
                        "args": getattr(tool, 'tool_args', {}),
                        "result": result,
                    }

            # Streaming text
            elif event_type == 'RunContent':
                content = getattr(event, 'content', None)
                if content:
                    yield {"type": "delta", "content": content}

            # Final content
            elif event_type == 'RunCompleted':
                content = getattr(event, 'content', None)
                if content:
                    yield {"type": "content", "content": content}

        yield {"type": "done"}

    except Exception as e:
        debug_log(f"Error: {e}")
        yield {"type": "error", "error": str(e)}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream the response with tool calls visible"""

    async def generate():
        loop = asyncio.get_event_loop()

        # Run the sync generator in a thread
        def get_events():
            return list(run_stream_sync(request.message))

        # Actually stream by yielding as we get events
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
            # Check queue with timeout to allow async
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
    """Non-streaming endpoint that returns full tool call details"""
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
            elif event["type"] == "content":
                final_content = event["content"]

        return {
            "response": final_content,
            "tool_calls": tool_calls,
            "session_id": request.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Dash API server...")
    print("📍 API docs: http://localhost:8000/docs")
    print("💬 Chat endpoint: POST http://localhost:8000/chat")
    print("📡 Stream endpoint: POST http://localhost:8000/chat/stream\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
