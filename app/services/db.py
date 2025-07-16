from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.postgre_models import Base, QueryHistory, Checkpoint
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DATABASE_URL = "postgresql://postgres:12345678@localhost:5432/test_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

model = SentenceTransformer("all-MiniLM-L6-v2")

def save_query_history(question: str, answer: str, user_id: int = 0):
    db = SessionLocal()
    db.add(QueryHistory(user_id=user_id, question=question, answer=answer))
    db.commit()
    db.close()

def save_checkpoint(step: str, data: dict, user_id: int = 0):
    db = SessionLocal()
    db.add(Checkpoint(user_id=user_id, step_name=step, state_data=data))
    db.commit()
    db.close()

def get_similar_history(query: str, user_id: int = 0, threshold=0.90):
    db = SessionLocal()
    histories = db.query(QueryHistory).filter_by(user_id=user_id).all()
    db.close()

    if not histories:
        return None

    query_vec = model.encode([query])
    question_vecs = model.encode([h.question for h in histories])
    sims = cosine_similarity(query_vec, question_vecs)[0]
    best_idx = np.argmax(sims)

    if sims[best_idx] > threshold:
        return histories[best_idx].answer
    return None

def get_recent_history(user_id: int = 0, max_turns: int = 5):
    db = SessionLocal()
    history = (
        db.query(QueryHistory)
        .filter_by(user_id=user_id)
        .order_by(QueryHistory.timestamp.desc())
        .limit(max_turns)
        .all()
    )
    db.close()
    return [(h.question, h.answer) for h in reversed(history)]
