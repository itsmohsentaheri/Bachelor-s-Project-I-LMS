import pytest
from sqlalchemy import text
from bachelor_s_project_i_lms.db.session import engine

def test_database_connection():
    """Gate 1: database is reachable."""
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1;")).scalar() == 1

def test_pgvector_enabled():
    """Gate 1: pgvector extension exists."""
    with engine.connect() as conn:
        ext = conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        ).scalar()
    assert ext == "vector", "pgvector is not enabled."

def test_chunks_have_embeddings():
    """Gate 2: at least one chunk is embedded."""
    with engine.connect() as conn:
        filled = conn.execute(
            text("SELECT COUNT(embedding) FROM book_chunks;")
        ).scalar()
    assert filled > 0, "No embeddings found in book_chunks."

def test_vector_search_returns_results():
    """Gate 2: vector search returns related chunks."""
    from bachelor_s_project_i_lms.scripts.vector_search import embed
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT embedding FROM book_chunks WHERE embedding IS NOT NULL LIMIT 1;")
        ).scalar()

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT chunk_id
                FROM book_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :q
                LIMIT 3;
            """),
            {"q": row},
        ).scalars().all()

    assert len(rows) > 0, "Vector search returned no rows."
