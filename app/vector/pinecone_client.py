import pinecone
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
INDEX_NAME = os.getenv("PINECONE_INDEX", "textile-products")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

def get_index():
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=1536,  # OpenAI's embedding size; adjust if using another model
            metric="cosine"
        )
    return pinecone.Index(INDEX_NAME)
