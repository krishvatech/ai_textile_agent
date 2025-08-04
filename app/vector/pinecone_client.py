from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "textile-products")

# Initialize Pinecone client
pinecone = Pinecone(api_key=PINECONE_API_KEY)

def get_index():
    # Get list of existing indexes
    existing_indexes = [idx["name"] for idx in pinecone.list_indexes()]
    
    if INDEX_NAME in existing_indexes:
        # If index exists, just connect to it without deleting
        print(f"Index '{INDEX_NAME}' already exists. Connecting to it.")
    else:
        # Create new index if it doesn't exist
        print(f"Creating new index '{INDEX_NAME}'.")
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    # Return the index object
    return pinecone.Index(INDEX_NAME)

if __name__ == "__main__":
    index = get_index()
    print(f"Successfully got index: {index}")