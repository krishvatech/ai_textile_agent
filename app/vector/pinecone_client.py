# import pinecone
# import os

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
# INDEX_NAME = os.getenv("PINECONE_INDEX", "textile-products")

# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# def get_index():
#     if INDEX_NAME not in pinecone.list_indexes():
#         pinecone.create_index(
#             name=INDEX_NAME,
#             dimension=1536,
#             metric="cosine"
#         )
#     return pinecone.Index(INDEX_NAME)
from pinecone import Pinecone, ServerlessSpec
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
INDEX_NAME = os.getenv("PINECONE_INDEX", "textile-products")

pc = Pinecone(api_key=PINECONE_API_KEY)

def get_index():
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="gcp",
                region=PINECONE_ENV
            )
        )
    return pc.Index(INDEX_NAME)

