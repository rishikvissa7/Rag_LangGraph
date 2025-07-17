# Import the Qdrant client to connect to the vector database
from qdrant_client import QdrantClient

# Import the SentenceTransformer to convert questions into vector embeddings
from sentence_transformers import SentenceTransformer

# Import os to read environment variables
import os

QDRANT_URL = os.getenv("https://77330263-0170-4417-8b94-4921a9edafd8.us-east4-0.gcp.cloud.qdrant.io")  
QDRANT_API_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ZW66wbYvoZwt1135PkcYNBN-4ysxptHAxX2XqSTORTE")  

# Create a client that connects to your Qdrant Cloud instance
client = QdrantClient(
    url=QDRANT_URL,          # URL of your Qdrant instance
    api_key=QDRANT_API_KEY   # API key for authentication
)

# Load the sentence embedding model which converts questions to vectors
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define a function to search for similar questions in a Qdrant collection
def search_collection(collection_name: str, query: str, top_k: int = 5):
    # Convert the user's question into a vector using the model
    vector = model.encode(query)

    # Search in the given collection using the query vector
    hits = client.search(
        collection_name=collection_name,  # Which collection to search in
        query_vector=vector,              # The vectorized user query
        limit=top_k                       # Number of top results to return
    )

    # Return a list of matched questions and answers with their similarity score
    return [
        {
            "question": h.payload["question"],  # The original question stored
            "answer": h.payload["answer"],      # The answer for that question
            "score": h.score                    # How similar it is to the input query
        }
        for h in hits
    ]
