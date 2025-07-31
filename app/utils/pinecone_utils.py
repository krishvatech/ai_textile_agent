import openai
import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone and OpenAI once at top of file
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX"])
openai.api_key = os.environ["GPT_API_KEY"]

async def query_products(user_query: str, lang: str = "en", top_k: int = 3):
    # Get semantic embedding of user query
    embedding_response = openai.embeddings.create(
        input=user_query,
        model="text-embedding-ada-002"  # or your preferred model
    )
    query_vec = embedding_response.data[0].embedding

    # Query Pinecone for top matches
    result = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    products = []
    for match in result.matches:
        meta = match.metadata
        products.append({
            "id": meta.get("id"),
            "category": meta.get("category"),
            "color": meta.get("color"),
            "description": meta.get("description"),
            "fabric": meta.get("fabric"),
            "price": meta.get("price"),
            "product_name": meta.get("product_name"),   # or adapt as needed
            "size": meta.get("size"),
        })
    return products
