from pinecone import Pinecone, ServerlessSpec
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
INDEX_NAME = os.getenv("PINECONE_INDEX", "textile-products")

pinecone = Pinecone(api_key=PINECONE_API_KEY)

def get_index():
    # Delete if exists
    existing_indexes = [idx["name"] for idx in pinecone.list_indexes()]
    if INDEX_NAME in existing_indexes:
        pinecone.delete_index(INDEX_NAME)

    # Create new index with correct dimension
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
