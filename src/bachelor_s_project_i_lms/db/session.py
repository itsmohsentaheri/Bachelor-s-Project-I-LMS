import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "secret-26e2971aql://postgres:postgres@localhost:5432/tutor_db")

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    print("\n🔍 Checking database connection and 'book_chunks' table status...\n")
    try:
        with engine.connect() as conn:
            # 1. Test basic connection
            res = conn.execute(text("SELECT 1;")).scalar()
            print("✅ 1. Database connection established successfully.")

            # 2. Check pgvector extension
            ext = conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            ).scalar()
            if ext == "vector":
                print("✅ 2. 'pgvector' extension is enabled.")
            else:
                print("❌ 2. 'pgvector' extension is NOT enabled! Run: CREATE EXTENSION vector;")

            # 3. Check row count and embeddings in book_chunks
            stats = conn.execute(text("""
                SELECT 
                    COUNT(*) AS total,
                    COUNT(embedding) AS filled_embeddings
                FROM book_chunks;
            """)).mappings().one()

            print(f"✅ 3. Total chunks in 'book_chunks': {stats['total']}")
            print(f"📊 Chunks with embeddings: {stats['filled_embeddings']}")

            # 4. Get vector dimensions if embeddings exist
            if stats['filled_embeddings'] > 0:
                dims = conn.execute(text("""
                    SELECT vector_dims(embedding) 
                    FROM book_chunks 
                    WHERE embedding IS NOT NULL 
                    LIMIT 1;
                """)).scalar()
                print(f"📐 Registered embedding dimensions: {dims}")
            else:
                print("⚠️ No embeddings found in the 'embedding' column yet.")

            print("\n🎉 Phase 3 Gate check completed successfully!")

    except Exception as e:
        print("\n❌ Error connecting to database or running checks:")
        print(e)
