from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.postgre_models import Base, QueryHistory, Checkpoint
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity # For calculating cosine similarity between vectors
import numpy as np

# PostgreSQL connection string
DATABASE_URL = "postgresql://postgres:12345678@localhost:5432/test_db"

# Create SQLAlchemy engine and session factory
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create all tables defined in Base (QueryHistory, Checkpoint)
Base.metadata.create_all(bind=engine)

# Load a pre-trained sentence transformer for converting text to vectors
model = SentenceTransformer("all-MiniLM-L6-v2")

# Save user query and generated answer to the QueryHistory table
def save_query_history(question: str, answer: str, user_id: int = 0):
    db = SessionLocal()
    db.add(QueryHistory(user_id=user_id, question=question, answer=answer))
    db.commit()
    db.close()

# Save intermediate step data (like LangGraph state) to Checkpoint table
def save_checkpoint(step: str, data: dict, user_id: int = 0):
    db = SessionLocal()
    db.add(Checkpoint(user_id=user_id, step_name=step, state_data=data))
    db.commit()
    db.close()

# Retrieve the most similar past answer to the given query using cosine similarity
def get_similar_history(query: str, user_id: int = 0, threshold=0.90):
    db = SessionLocal()
    
    # Get all previous query-answer records for the user
    histories = db.query(QueryHistory).filter_by(user_id=user_id).all()
    db.close()

    if not histories:
        return None

    # Convert query and history questions to embeddings
    query_vec = model.encode([query])
    question_vecs = model.encode([h.question for h in histories])

    # Calculate cosine similarity between query and all past questions
    sims = cosine_similarity(query_vec, question_vecs)[0]
    best_idx = np.argmax(sims)

    # If similarity is above the threshold, return the matching past answer
    if sims[best_idx] > threshold:
        return histories[best_idx].answer
    return None

# Get the most recent `max_turns` (default 5) question-answer pairs for a user
def get_recent_history(user_id: int = 0, max_turns: int = 5):
    db = SessionLocal()
    
    # Fetch recent history ordered by latest timestamp
    history = (
        db.query(QueryHistory)
        .filter_by(user_id=user_id)
        .order_by(QueryHistory.timestamp.desc())
        .limit(max_turns)
        .all()
    )
    db.close()

    # Return the history in chronological order (oldest to newest)
    return [(h.question, h.answer) for h in reversed(history)]
