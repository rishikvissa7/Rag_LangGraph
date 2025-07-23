# app/services/postgres_storage.py
# Database operations for conversation history and checkpoints

import json
from app.db.db import SessionLocal
from app.models.models import ConversationTurn, Checkpoint, Conversation
from datetime import datetime
import logging
from sqlalchemy.orm import Session

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def save_turn(user_id: str, role: str, content: str, session_id: str = "default_session"):
    db: Session = SessionLocal()

    # 1. Check or create conversation for user + session
    conversation = db.query(Conversation).filter_by(user_id=user_id, session_id=session_id).first()
    if not conversation:
        conversation = Conversation(user_id=user_id, session_id=session_id, timestamp=datetime.utcnow())
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # 2. Insert new turn with conversation_id
    turn = ConversationTurn(
        conversation_id=conversation.id,
        role=role,
        content=content,
        timestamp=datetime.utcnow()
    )
    db.add(turn)
    db.commit()
    db.close()


def get_history(user_id: str = "default_user", limit: int = 50) -> list:
    """Get conversation history for a user."""
    db = SessionLocal()
    try:
        turns = db.query(ConversationTurn).filter(
            ConversationTurn.user_id == user_id
        ).order_by(ConversationTurn.timestamp.asc()).limit(limit).all()
        
        history = []
        for turn in turns:
            history.append({
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp
            })
        
        logger.info(f"Retrieved {len(history)} conversation turns for {user_id}")
        return history
    except Exception as e:
        logger.error(f"Failed to load history: {e}")
        return []
    finally:
        db.close()

def save_checkpoint(user_id: str, state: dict):
    """Save checkpoint state for a user."""
    db = SessionLocal()
    try:
        checkpoint = db.query(Checkpoint).filter(Checkpoint.user_id == user_id).first()
        if checkpoint:
            checkpoint.state = state
            checkpoint.updated_at = datetime.utcnow()
        else:
            checkpoint = Checkpoint(user_id=user_id, state=state)
            db.add(checkpoint)
        db.commit()
        db.save(checkpoint)
        logger.info(f"Checkpoint saved for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        db.rollback()
    finally:
        db.close()

def load_checkpoint(user_id: str = "default_user") -> dict:
    """Load the last saved checkpoint for a user."""
    db = SessionLocal()
    try:
        cp = db.query(Checkpoint).filter_by(user_id=user_id).first()
        if cp:
            logger.info(f"Loaded checkpoint for user {user_id}")
            return cp.state
        logger.info(f"No checkpoint found for user {user_id}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        return {}
    finally:
        db.close()

# Session-based functions for workflow compatibility
def get_session_history(session_id: str) -> dict:
    """Get conversation history by session ID."""
    db = SessionLocal()
    try:
        record = db.query(Conversation).filter(Conversation.session_id == session_id).first()
        if record:
            return json.loads(record.state) if isinstance(record.state, str) else record.state
        else:
            return {}
    except Exception as e:
        logger.error(f"Failed to load session history: {e}")
        return {}
    finally:
        db.close()

def save_session_checkpoint(session_id: str, state: dict):
    """Save checkpoint state for a session."""
    db = SessionLocal()
    try:
        conversation_json = json.dumps(state) if not isinstance(state, str) else state
        record = db.query(Conversation).filter(Conversation.session_id == session_id).first()
        if record:
            record.state = json.dumps(state)
            record.updated_at = datetime.utcnow()
        else:
            record = Conversation(session_id=session_id, state=json.dumps(state))
            db.add(record)
        db.commit()
        logger.info(f"Session checkpoint saved for {session_id}")
    except Exception as e:
        logger.error(f"Failed to save session checkpoint: {e}")
        db.rollback()
    finally:
        db.close()