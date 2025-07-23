# app/services/upload_to_qdrant.py
# Script to upload JSON document data to Qdrant company_info collection

import json
import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def initialize_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client with environment variables."""
    try:
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        logger.info("Qdrant client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {str(e)}")
        raise

def ensure_collection_exists(client: QdrantClient, collection_name: str = "company_info"):
    """Ensure the company_info collection exists with the correct configuration."""
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if collection_name not in collection_names:
            logger.info(f"Creating Qdrant collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,  # Matches all-MiniLM-L6-v2 embedding size
                    distance=Distance.COSINE
                ),
                optimizers_config={"indexing_threshold": 0}  # Index all vectors
            )
        # Ensure keyword index on 'company' field
        indexes = client.get_payload_schema(collection_name=collection_name)
        has_keyword_index = any(
            index.field_name == "company" and index.field_type == "keyword"
            for index in indexes.indexes
        ) if hasattr(indexes, "indexes") else False
        if not has_keyword_index:
            logger.info("Creating keyword index for 'company' field")
            client.create_payload_index(
                collection_name=collection_name,
                field_name="company",
                field_schema="keyword"
            )
            logger.info("Created keyword index for 'company' field")
        else:
            logger.info("Keyword index for 'company' field already exists")
    except Exception as e:
        logger.error(f"Failed to initialize collection {collection_name}: {str(e)}")
        raise

def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON data must be a list of dictionaries")
        for item in data:
            if not all(key in item for key in ["company", "info"]):
                raise ValueError("Each JSON object must contain 'company' and 'info' keys")
        logger.info(f"Loaded {len(data)} records from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON data from {file_path}: {str(e)}")
        raise

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using SentenceTransformer."""
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(texts, show_progress_bar=True).tolist()
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise

def upload_to_qdrant(json_file_path: str, collection_name: str = "company_info"):
    """Upload JSON data to Qdrant company_info collection."""
    try:
        # Initialize Qdrant client and collection
        client = initialize_qdrant_client()
        ensure_collection_exists(client, collection_name)

        # Load JSON data
        data = load_json_data(json_file_path)

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
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info(f"Successfully uploaded {len(points)} points to {collection_name}")
    except Exception as e:
        logger.error(f"Failed to upload data to Qdrant: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python upload_to_qdrant.py <json_file_path>")
        sys.exit(1)
    json_file_path = sys.argv[1]
    upload_to_qdrant(json_file_path)