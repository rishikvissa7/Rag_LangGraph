# app/api/data_upload.py
# FastAPI router for uploading JSON data to Qdrant

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer
import json
import logging
import os
from dotenv import load_dotenv
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

router = APIRouter()

# Initialize SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("SentenceTransformer initialized successfully")

def load_json_data(file_content: bytes, filename: str) -> List[Dict]:
    """Load and validate JSON data from uploaded file, transforming to Qdrant format."""
    try:
        data = json.loads(file_content.decode('utf-8'))
        if not isinstance(data, list):
            raise ValueError("JSON data must be a list of dictionaries")
        for item in data:
            if not all(key in item for key in ["question", "answer"]):
                raise ValueError("Each JSON object must contain 'question' and 'answer' keys")
        # Extract company name from filename (e.g., Tcs.json -> TCS)
        company_name = os.path.splitext(filename)[0].capitalize()
        # Transform to Qdrant-compatible format: {"company": company_name, "info": answer}
        transformed_data = [
            {"company": company_name, "info": item["answer"]}
            for item in data
        ]
        logger.info(f"Loaded and transformed {len(transformed_data)} records from {filename}")
        return transformed_data
    except Exception as e:
        logger.error(f"Failed to load JSON data from {filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON data: {str(e)}")

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using SentenceTransformer."""
    try:
        embeddings = embedder.encode(texts, show_progress_bar=True).tolist()
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@router.post("/upload-data")
async def upload_to_qdrant(file: UploadFile = File(...), qdrant_client: QdrantClient = Depends(lambda: QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
))):
    """Upload JSON data to Qdrant company_info collection."""
    try:
        # Validate file type
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="File must be a JSON file")

        # Read file content
        file_content = await file.read()
        data = load_json_data(file_content, file.filename)

        # Prepare points for Qdrant
        texts = [item["info"] for item in data]
        embeddings = generate_embeddings(texts)
        points = [
            PointStruct(
                id=index,
                vector=embedding,
                payload={
                    "company": item["company"],
                    "info": item["info"]
                }
            )
            for index, (item, embedding) in enumerate(zip(data, embeddings))
        ]

        # Upsert points to Qdrant
        qdrant_client.upsert(
            collection_name="company_info",
            points=points
        )
        logger.info(f"Successfully uploaded {len(points)} points to company_info from {file.filename}")
        return {"message": f"Successfully uploaded {len(points)} points to Qdrant"}
    except Exception as e:
        logger.error(f"Failed to upload data to Qdrant: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload data to Qdrant: {str(e)}")