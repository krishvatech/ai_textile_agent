
from typing import Dict, Any, List, Optional
import os
from dotenv import load_dotenv
load_dotenv()

# Fallback Pinecone direct
from app.core.product_search import pinecone_fetch_records
from app.core.rental_utils import is_variant_available

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "textile-products")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ns1")
USE_LLAMA_PRIMARY = os.getenv("USE_LLAMAINDEX_PRIMARY", "1") not in ("0","false","False")

# Best-effort imports for LlamaIndex; if missing we'll fallback.
try:
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from pinecone import Pinecone as PcClient
    LLAMA_OK = True
except Exception:
    LLAMA_OK = False

def _entities_to_query(text: str, entities: Dict[str, Any]) -> str:
    parts = [text or ""]
    for k in ("category","type","fabric","color","size","occasion"):
        v = entities.get(k)
        if v:
            if isinstance(v, list):
                parts.append(f"{k}: " + ", ".join(map(str, v)))
            else:
                parts.append(f"{k}: {v}")
    # Budget hint improves semantic retrieval sometimes
    budget = entities.get("budget") or entities.get("max_price")
    if budget: parts.append(f"budget <= {budget}")
    return " | ".join(parts)

async def tool_product_search(entities: Dict[str, Any], tenant_id: int, text: Optional[str]=None) -> List[Dict[str, Any]]:
    """Primary retriever: LlamaIndex (Pinecone). Fallback to pinecone_fetch_records."""
    if USE_LLAMA_PRIMARY and LLAMA_OK:
        try:
            pc = PcClient(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(PINECONE_INDEX)
            vs = PineconeVectorStore(index=index, namespace=PINECONE_NAMESPACE)
            storage_context = StorageContext.from_defaults(vector_store=vs)
            idx = VectorStoreIndex.from_vector_store(vs, storage_context=storage_context)
            retriever = idx.as_retriever(similarity_top_k=12)
            query = _entities_to_query(text or "", entities or {})
            # llama-index retriever is sync; run directly
            nodes = retriever.retrieve(query)
            out: List[Dict[str, Any]] = []
            for n in nodes:
                md = getattr(n.node, "metadata", {}) or {}
                score = getattr(n, "score", None)
                out.append({
                    "id": md.get("id") or md.get("product_id"),
                    "score": score,
                    "name": md.get("name"),
                    "is_rental": md.get("is_rental"),
                    "category": md.get("category"),
                    "occasion": md.get("occasion"),
                    "fabric": md.get("fabric"),
                    "color": md.get("color"),
                    "size": md.get("size"),
                    "variant_id": md.get("variant_id"),
                    "product_id": md.get("product_id"),
                    "image_url": md.get("image_url"),
                    "product_url": md.get("product_url"),
                    "price": md.get("price"),
                    "available_stock": md.get("available_stock"),
                    "type": md.get("type"),
                    "tenant_id": md.get("tenant_id"),
                })
            # Deduplicate by variant_id or product_id preserving order
            seen = set()
            uniq = []
            for it in out:
                key = it.get("variant_id") or ("p:"+str(it.get("product_id")))
                if key and key not in seen:
                    seen.add(key)
                    uniq.append(it)
            return uniq
        except Exception:
            # fall back below
            pass
    # Fallback: direct Pinecone search implemented by your code
    return await pinecone_fetch_records(entities=entities or {}, tenant_id=tenant_id)

async def tool_availability_filter(db, products: List[Dict[str, Any]], start_date, end_date=None) -> List[Dict[str, Any]]:
    """Keep only items whose variant_id is available for the requested dates. If no variant_id, keep as-is."""
    if not products or not db:
        return products or []
    kept = []
    for item in products:
        vid = item.get("variant_id")
        if not vid:
            kept.append(item)
            continue
        try:
            ok = await is_variant_available(db, int(vid), start_date, end_date)
        except Exception:
            ok = True  # be permissive on DB failure
        if ok:
            kept.append(item)
    return kept

def tool_rerank(products: List[Dict[str, Any]], entities: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not products:
        return products
    max_price = None
    budget = entities.get("budget") or entities.get("max_price")
    try:
        if budget is not None:
            max_price = float(budget)
    except Exception:
        max_price = None
    desired_occ = (entities.get("occasion") or "").strip().lower() if isinstance(entities.get("occasion"), str) else None

    def score_item(it: Dict[str, Any]) -> float:
        base = float(it.get("score") or 0.0)
        price = None
        try:
            price = float(it.get("price")) if it.get("price") is not None else None
        except Exception:
            price = None
        # Budget preference: boost if <= budget, small penalty if above
        if max_price is not None and price is not None:
            if price <= max_price:
                base += 0.2
            else:
                base -= 0.2
        # Occasion match small boost
        if desired_occ:
            p_occ = (it.get("occasion") or "").strip().lower()
            if p_occ and desired_occ in p_occ:
                base += 0.1
        return base

    return sorted(products, key=score_item, reverse=True)[:10]
