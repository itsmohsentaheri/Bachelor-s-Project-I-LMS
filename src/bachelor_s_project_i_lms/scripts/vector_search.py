from bachelor_s_project_i_lms.db.session import SessionLocal
from bachelor_s_project_i_lms.db.models import BookChunk

def embed(text: str) -> list[float]:
    """Use the same embedding function as in embed_chunks.py."""
    raise NotImplementedError("Plug in your embedding model here.")

def search(query: str, k: int = 3):
    query_vec = embed(query)
    db = SessionLocal()
    try:
        results = (
            db.query(BookChunk)
            .order_by(BookChunk.embedding.cosine_distance(query_vec))
            .limit(k)
            .all()
        )
        print(f"\n🔎 Query: {query}")
        for rank, c in enumerate(results, start=1):
            preview = (c.content or "")[:160].replace("\n", " ")
            print(f"\n[{rank}] chunk_id={c.chunk_id} | page={c.page_number} | source={c.source_title}")
            print(f"    {preview}...")

        return results
    finally:
        db.close()

if __name__ == "__main__":
    search("a sentence related to one of your chapter titles")
