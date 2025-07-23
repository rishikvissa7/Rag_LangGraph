# app/main.py
# FastAPI application entry point

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from app.api.query_router import router as query_router
from app.api.data_upload import router as upload_router
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType
from app.db.db import Base, engine
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG Multi-Agent System", version="1.0.0")

# Initialize Qdrant client
try:
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    logger.info("Qdrant client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant client: {str(e)}")
    qdrant_client = None

@app.on_event("startup")
async def startup_event():
    """Initialize database tables and Qdrant collection on startup."""
    
    # Create database tables
    try:
        # Import models to ensure they're registered with Base
        
        # Create all tables
        # (re-run app.main or manually run)
        Base.metadata.create_all(bind=engine)

        logger.info("✅ Database tables created successfully")
        
        # Verify tables were created
        from sqlalchemy import inspect
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        logger.info(f"📋 Created tables: {table_names}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create database tables: {e}")
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")

    # Initialize Qdrant collection
    if qdrant_client:
        try:
            collections = qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if "company_info" not in collection_names:
                logger.info("Creating Qdrant collection: company_info")
                qdrant_client.create_collection(
                    collection_name="company_info",
                    vectors_config=VectorParams(
                        size=384,  # Matches all-MiniLM-L6-v2 embedding size
                        distance=Distance.COSINE
                    ),
                    optimizers_config={"indexing_threshold": 0}  # Index all vectors
                )
                logger.info("✅ Qdrant collection 'company_info' created")
            else:
                logger.info("✅ Qdrant collection 'company_info' already exists")
            
            # Check for keyword index on 'company' field
            try:
                indexes = qdrant_client.get_payload_schema(collection_name="company_info")
                has_keyword_index = any(
                    index.field_name == "company" and index.field_type == "keyword"
                    for index in indexes.indexes
                ) if hasattr(indexes, "indexes") and indexes.indexes else False
                
                if not has_keyword_index:
                    logger.info("Creating keyword index for 'company' field")
                    qdrant_client.create_payload_index(
                        collection_name="company_info",
                        field_name="company",
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    logger.info("✅ Created keyword index for 'company' field")
                else:
                    logger.info("✅ Keyword index for 'company' field already exists")
                    
            except Exception as index_error:
                logger.warning(f"⚠️ Could not verify/create keyword index: {index_error}")
                # Try to create index anyway
                try:
                    qdrant_client.create_payload_index(
                        collection_name="company_info",
                        field_name="company",
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    logger.info("✅ Created keyword index for 'company' field (fallback)")
                except Exception as fallback_error:
                    logger.error(f"❌ Failed to create keyword index: {fallback_error}")
                    
        except Exception as e:
            logger.error(f"❌ Qdrant collection initialization failed: {str(e)}")
            # Don't raise exception here, allow app to start without Qdrant if needed
            logger.warning("⚠️ Application starting without Qdrant functionality")
    else:
        logger.warning("⚠️ Qdrant client not available, skipping collection initialization")
# Register API routes
app.include_router(query_router, prefix="/api", tags=["Query"])
app.include_router(upload_router, prefix="/api/data", tags=["Data Upload"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)