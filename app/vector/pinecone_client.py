from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "textile-products")
IMAGE_INDEX_NAME = os.getenv("PINECONE_IMAGE_INDEX", "textile-products-image")

pinecone = Pinecone(api_key=PINECONE_API_KEY)

def get_index(index_name=INDEX_NAME):
    """Connect to an existing Pinecone index by name."""
    return pinecone.Index(index_name)

def get_image_index():
    """Connect to the existing image index."""
    image_index_name = os.getenv("PINECONE_IMAGE_INDEX", "textile-products-image")
    return pinecone.Index(image_index_name)

if __name__ == "__main__":
    # Just connect (not create!) to ensure indexes are live
    try:
        text_index = get_index()
        print(f"Text index connected: {text_index}")
    except Exception as e:
        print(f"Failed to connect to text index: {e}")

    try:
        image_index = get_image_index()
        print(f"Image index connected: {image_index}")
    except Exception as e:
        print(f"Failed to connect to image index: {e}")

    print("Available indexes:", [i['name'] for i in pinecone.list_indexes()])
