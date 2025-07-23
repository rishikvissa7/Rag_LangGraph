# app/services/qdrant_search.py
# Qdrant search functionality

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Qdrant client
try:
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    logger.info("Qdrant client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant client: {str(e)}")
    raise

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def qdrant_search(query: str, company: str, top_k: int = 1) -> str:
    """Search Qdrant company_info collection for company-specific information."""
    try:
        # Convert query to vector
        vector = model.encode(query).tolist()
        logger.info(f"Executing Qdrant search for query: {query}, company: {company}")

        # Check for company index
        collection_info = client.get_collection(collection_name="company_info")
        indexes = collection_info.payload_indexes or []
        if not any(index.field_name == "company" and index.field_schema == "keyword" for index in indexes):
            logger.warning("Keyword index for 'company' not found")
            return "Error: Qdrant index for 'company' field is missing."

        # Perform search with company filter
        hits = client.search(
            collection_name="company_info",
            query_vector=vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="company",
                        match=dict(value=company)
                    )
                ]
            ),
            limit=top_k,
            with_payload=True
        )

        logger.info(f"Qdrant search result: {hits}")
        if hits:
            return hits[0].payload.get("info", "No relevant information found.")
        return "No information found for this company."
    except Exception as e:
        logger.error(f"Qdrant search failed for {company}: {str(e)}")
        return f"Error searching Qdrant: {str(e)}"