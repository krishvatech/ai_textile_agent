from app.vector.pinecone_client import get_index
import openai
import os
from app.db.models import Product

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

async def find_similar_products(db, tenant_id, color=None, product_type=None, price_range=None, top_k=3):
    # Use vector search here; fallback to SQL filter for demo
    query = db.query(Product).filter(Product.tenant_id == tenant_id)
    if color:
        query = query.filter(Product.color.ilike(f"%{color}%"))
    if product_type:
        query = query.filter(Product.type.ilike(f"%{product_type}%"))
    if price_range:
        min_p, max_p = map(float, price_range.split("-"))
        query = query.filter(Product.price.between(min_p, max_p))
    products = await query.limit(top_k).all()
    # Convert to dict/list for WhatsApp/media sending
    return [p.as_dict() for p in products]
