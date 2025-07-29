import os, asyncio
import openai
from app.db.models import Product
from app.db.session import SessionLocal
from app.vector.pinecone_client import get_index

openai.api_key = os.getenv("GPT_API_KEY")

async def embed_products_for_tenant(tenant_id):
    index = get_index()
    async with SessionLocal() as session:
        prods = await session.execute(
            "SELECT * FROM products WHERE tenant_id = :tid", {"tid": tenant_id}
        )
        for prod in prods.fetchall():
            prod = prod[0] if isinstance(prod, tuple) else prod
            text = f"{prod.name}. {prod.description}. Color: {prod.color}. Price: {prod.price}"
            embedding = openai.embeddings.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]
            index.upsert([(f"{tenant_id}_{prod.id}", embedding, {"tenant_id": tenant_id, "product_id": prod.id})])
            print(f"Upserted: {prod.name}")

if __name__ == "__main__":
    tenant_id = int(input("Tenant ID? "))
    asyncio.run(embed_products_for_tenant(tenant_id))
