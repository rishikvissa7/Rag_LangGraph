import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL")
if not POSTGRES_URL:
    raise ValueError("POSTGRES_URL environment variable is not set!")

engine = create_engine(POSTGRES_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
