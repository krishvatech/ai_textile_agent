
"""LlamaIndex ingestion & query setup (skeleton).
This uses your existing DB models to build Documents and backs them with Pinecone.
"""
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
load_dotenv()

# Import guarded to avoid ImportError at import-time if not installed
try:
    from llama_index.core import Document, VectorStoreIndex, StorageContext
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from pinecone import Pinecone as PcClient
except Exception:  # pragma: no cover
    Document = None
    VectorStoreIndex = None
    StorageContext = None
    PineconeVectorStore = None
    PcClient = None

from app.db.session import SessionLocal
from app.db import models

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "textile-products")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ns1")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def _to_documents(rows: List[Dict[str, Any]]) -> List['Document']:
    docs = []
    for r in rows:
        text = f"""{r.get('name','')}
Color: {r.get('color','')}
Fabric: {r.get('fabric','')}
Category: {r.get('category','')}
Type: {r.get('type','')}
Price: {r.get('price','')}
Rental: {r.get('is_rental','')}
Size: {r.get('size','')}
"""
        meta = {k:v for k,v in r.items() if k not in {'id','name'}}
        docs.append(Document(text=text, metadata=meta, doc_id=str(r.get("id"))))
    return docs

async def ingest_tenant_catalog(tenant_id: int) -> Dict[str, Any]:
    if VectorStoreIndex is None:
        return {"ok": False, "message": "LlamaIndex extras not installed."}

    # 1) Load products/variants
    async with SessionLocal() as db:
        q = await db.execute(models.Product.__table__.select().where(models.Product.tenant_id == tenant_id))
        products = [dict(row) for row in q.mappings().all()]

    # 2) Convert to Documents
    docs = _to_documents(products)

    # 3) Build Pinecone vector store
    pc = PcClient(api_key=PINECONE_API_KEY)
    vs = PineconeVectorStore(pc.Index(PINECONE_INDEX), namespace=PINECONE_NAMESPACE)
    storage_context = StorageContext.from_defaults(vector_store=vs)

    # 4) Index
    _ = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    return {"ok": True, "inserted": len(docs)}

