from bachelor_s_project_i_lms.db.session import SessionLocal
from bachelor_s_project_i_lms.db.models import BookChunk

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"  # 1024 dimensions, must match chunk.ipynb

_model = None  # lazy singleton: load once per process

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def embed(text: str) -> list[float]:
    """Same embedding config as in chunk.ipynb (bge-large-en-v1.5, normalized)."""
    vec = _get_model().encode([text], normalize_embeddings=True)[0]
    return [float(x) for x in vec]

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
