"""One-off script: re-title all conversations using Claude Haiku."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anthropic
from sqlalchemy import create_engine, text
from db.url import db_url

engine = create_engine(db_url)
client = anthropic.Anthropic()

# Fetch all conversations
with engine.connect() as conn:
    rows = conn.execute(text("SELECT id, title, messages FROM conversations ORDER BY updated_at DESC")).fetchall()

print(f"Found {len(rows)} conversations\n")

for row in rows:
    conv_id, old_title, messages = str(row[0]), row[1], row[2] or []

    first_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), None)
    if not first_msg:
        print(f"  SKIP {conv_id[:8]} — no user message")
        continue

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            messages=[{"role": "user", "content": f"Title this chat: {first_msg[:200]}"}],
            system="You are a title generator. Output 3-6 words, plain text only. No markdown, no bold, no asterisks, no quotes, no colons, no punctuation. Just the words. Examples: Stock Price Analysis for AMZN, F1 Championship Winners by Decade, TidyTuesday Water Access Data",
        )
        new_title = resp.content[0].text.strip().strip('"\'').replace('*', '').replace('#', '').replace(':', '')

        with engine.connect() as conn:
            conn.execute(text("UPDATE conversations SET title = :title WHERE id = :id"), {"title": new_title, "id": conv_id})
            conn.commit()

        print(f"  {(old_title or '(none)')[:40]:<42} -> {new_title}")

    except Exception as e:
        print(f"  ERROR {conv_id[:8]}: {e}")

print("\nDone.")
