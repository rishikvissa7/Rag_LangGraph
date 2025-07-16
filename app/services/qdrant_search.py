from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os

QDRANT_URL = os.getenv("https://77330263-0170-4417-8b94-4921a9edafd8.us-east4-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ZW66wbYvoZwt1135PkcYNBN-4ysxptHAxX2XqSTORTE")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

def search_collection(collection_name: str, query: str, top_k: int = 5):
    vector = model.encode(query)
    hits = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_k
    )
    return [{"question": h.payload["question"], "answer": h.payload["answer"], "score": h.score} for h in hits]
