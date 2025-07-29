from app.vector.pinecone_client import get_index
import openai
import os

openai.api_key = os.getenv("GPT_API_KEY")

def get_embeddings(text):
    resp = openai.embeddings.create(input=[text], model="text-embedding-ada-002")
    return resp["data"][0]["embedding"]

def search_products(query, tenant_id, top_k=3):
    embedding = get_embeddings(query)
    index = get_index()
    res = index.query(vector=embedding, top_k=top_k, include_metadata=True, filter={"tenant_id": {"$eq": tenant_id}})
    products = []
    for match in res["matches"]:
        meta = match["metadata"]
        products.append({
            "product_id": meta["product_id"],
            "score": match["score"]
        })
    return products
