from sqlalchemy import Column, Integer, String, Text
from pgvector.sqlalchemy import Vector
from bachelor_s_project_i_lms.db.session import Base

EMBEDDING_DIM = 1024  # Sync this with the embedding model you use.

class BookChunk(Base):
    __tablename__ = "book_chunks"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(String, unique=True, index=True)
    source_title = Column(String, nullable=True)
    page_number = Column(Integer, nullable=True)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
