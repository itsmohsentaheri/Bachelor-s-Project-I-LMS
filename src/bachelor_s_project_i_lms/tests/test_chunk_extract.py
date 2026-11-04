import os
from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict
import psycopg
from dotenv import load_dotenv

# Import project modules
# Note: Adjust the import path if needed based on your package structure
from bachelor_s_project_i_lms.llm.client import call_llm

# Load environment variables (.env)
load_dotenv()

# 1. Define expected output schema with alias support
class DecomposeOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    concepts: list[str] = Field(
        default_factory=list,
        alias="key_concepts",
        description="Extracted key concepts from the text"
    )
    summary: str = Field(
        ...,
        description="Short summary of the chunk content"
    )



def test_llm_decompose_integration():
    print("🚀 Starting LLM Decompose integration test...")

    # 2. Database connection to fetch a real chunk
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("❌ DATABASE_URL environment variable is not set!")

    print("🔌 Connecting to PostgreSQL / Supabase database...")
    with psycopg.connect(db_url) as conn:
        row = conn.execute(
            """
            SELECT id, chunk_id, source_title, page_number, content 
            FROM book_chunks 
            WHERE content IS NOT NULL AND length(trim(content)) > 0 
            LIMIT 1
            """
        ).fetchone()

        if not row:
            print("⚠️ No valid chunk found in the book_chunks table!")
            return

        chunk_db_id, chunk_id, source_title, page_number, content = row
        print(f"📖 Selected chunk: ID={chunk_id} | Source: {source_title} | Page: {page_number}")

    # 3. Format prompt into OpenAI-style chat messages
    user_prompt = (
        f"Analyze the following textbook excerpt. Extract key concepts and provide a short summary.\n"
        f"You must return a valid JSON object matching this schema exactly:\n"
        f'{{"concepts": ["concept 1", "concept 2"], "summary": "brief summary text"}}\n\n'
        f"Text excerpt:\n{content[:1500]}"
    )

    messages = [
        {"role": "user", "content": user_prompt}
    ]

    print("🤖 Sending request to call_llm...")

    # 4. Invoke call_llm with the correct signature
    try:
        result = call_llm(
            messages=messages,
            response_model=DecomposeOutput,
            system="You are an intelligent educational assistant specialized in curriculum decomposition and text analysis."
        )
        print("✅ Successfully received parsed response from LLM:")
        print(result)
    except Exception as e:
        print(f"❌ Error during call_llm execution: {e}")
        raise e

        # 5. Verify and check the event logging in the events table
    print("🔍 Checking database events table for the latest event log...")
    with psycopg.connect(db_url) as conn:
        event_row = conn.execute(
            """
            SELECT event_id, type, payload 
            FROM events 
            ORDER BY event_id DESC 
            LIMIT 1
            """
        ).fetchone()

        if not event_row:
            print("❌ Error: No event record found in the events table!")
            assert False, "Event not found in database"

        event_id, event_type, payload = event_row
        print(f"🎉 Event found! ID: {event_id} | Type: {event_type}")

        # 6. Validate payload structure
        assert payload is not None, "Payload is empty."
        print(f"📦 Logged payload keys: {list(payload.keys())}")
        
        print("✨ All tests passed successfully! LLM call and database logging verified.")


if __name__ == "__main__":
    test_llm_decompose_integration()
