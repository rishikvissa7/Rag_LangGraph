# app/models/models.py
import logging
from sqlalchemy import Column, ForeignKey, Integer, String, Text, DateTime, JSON
from datetime import datetime
from app.db.db import Base
from sqlalchemy.orm import relationship

logger = logging.getLogger(__name__)

class ConversationTurn(Base):
    __tablename__ = "conversation_turns"
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))  # ✅ link to Conversation
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="turns")  # ✅ bidirectional


class Checkpoint(Base):
    __tablename__ = "rag_checkpoints"
    user_id = Column(String(255), primary_key=True, default="default_user")
    state = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Fixed: Added the Conversation class that's imported in postgres_storage.py
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, index=True)
    user_id = Column(String(255), index=True)  # ✅ Needed for querying by user
    timestamp = Column(DateTime, default=datetime.utcnow)
    state = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    turns = relationship("ConversationTurn", back_populates="conversation", cascade="all, delete-orphan")  # ✅ bidirectional