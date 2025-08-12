import os
import asyncio
import logging
from functools import lru_cache
from typing import Dict, Any, List
import anyio
from dotenv import load_dotenv
from pinecone import Pinecone
import open_clip
import torch

# ---------- Setup ----------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TEXT_INDEX_NAME = os.getenv("PINECONE_INDEX", "textile-products")
NAMESPACE = os.getenv("PINECONE_NAMESPACE")

if not PINECONE_API_KEY:
    raise EnvironmentError("❌ PINECONE_API_KEY not set")

# Pinecone client (reuse)
pinecone = Pinecone(api_key=PINECONE_API_KEY)

# Logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# Device & Torch perf hints
device = "cpu"
try:
    torch.set_num_threads(min(4, (os.cpu_count() or 4)))
except Exception:
    pass

# ---- CLIP model (init once, reuse) ----
# Use quickgelu variant to match 'openai' weights
model_name = "ViT-B-32-quickgelu"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
tokenizer = open_clip.get_tokenizer(model_name)
model.eval()
model = model.to(device)

# Reuse index + query parameters
PC_INDEX = pinecone.Index(TEXT_INDEX_NAME)
TEXT_TOP_K = 5  # lower = faster; raise only if needed


def flexible_compare(a, b) -> bool:
    """Robust equality for mixed types (bool/str/num)."""
    if isinstance(a, bool) or isinstance(b, bool):
        def to_bool(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.strip().lower() in {"true", "yes", "1"}
            return bool(val)
        return to_bool(a) == to_bool(b)
    return str(a) == str(b)


def build_metadata_filter(tenant_id: int, entities_cap: dict) -> dict:
    """
    Build a Pinecone metadata filter. Only include fields you index as exact-match metadata.
    """
    f = {"tenant_id": {"$eq": tenant_id}}
    for key in ("category", "size", "color", "is_rental", "type"):
        val = entities_cap.get(key)
        if val not in [None, "", [], {}]:
            if isinstance(val, str):
                f[key] = {"$eq": val}
            elif isinstance(val, bool):
                f[key] = {"$eq": bool(val)}
            else:
                f[key] = {"$eq": val}
    return f


@lru_cache(maxsize=512)
def embed_text_cached(query_text: str) -> List[float]:
    """
    Cached CLIP text embedding (CPU).
    Returns a plain list[float] (NOT a numpy array) for Pinecone serialization.
    """
    tokens = tokenizer([query_text]).to(device)
    with torch.inference_mode():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    # ensure float32 and convert to Python list
    return feats[0].cpu().numpy().astype("float32").tolist()


async def pinecone_fetch_records(entities: dict, tenant_id: int) -> List[Dict[str, Any]]:
    """
    Fast product fetch:
      - Normalizes string entities (capitalize) to match your stored metadata style
      - Cached CLIP text embedding
      - Single Pinecone query with include_metadata=True (no extra fetch)
      - Server-side metadata filtering
      - Optional local flexible filter (cheap on <=5 items)
    """
    # Normalize (capitalize strings if your metadata is capitalized)
    entities_cap = {k: (str(v).capitalize() if isinstance(v, str) else v) for k, v in entities.items()}

    # Build compact text query from non-empty strings
    texts = [str(v) for v in entities_cap.values() if isinstance(v, str) and v.strip()]
    if not texts:
        logger.info("pinecone_fetch_records: no non-empty string fields in entities; returning [].")
        return []

    query_text = " ".join(texts)

    # Get (cached) embedding as list[float]
    query_vector = embed_text_cached(query_text)

    # Server-side metadata filter
    md_filter = build_metadata_filter(tenant_id, entities_cap)

    # Run the query in a worker thread (Pinecone client is sync)
    def _do_query():
        return PC_INDEX.query(
            vector=query_vector,          # list[float]
            top_k=TEXT_TOP_K,
            namespace=NAMESPACE,
            filter=md_filter,
            include_metadata=True,
            include_values=False,
        )

    response = await anyio.to_thread.run_sync(_do_query)

    # Build matches directly from returned metadata (single round-trip)
    matches: List[Dict[str, Any]] = []
    for m in getattr(response, "matches", []) or []:
        md = getattr(m, "metadata", {}) or {}
        item = {
            "id": getattr(m, "id", None),
            "score": getattr(m, "score", None),
            "product_name": md.get("product_name"),
            "is_rental": md.get("is_rental"),
            "category": md.get("category"),
            "occasion": md.get("occasion"),
            "fabric": md.get("fabric"),
            "color": md.get("color"),
            "size": md.get("size"),
            "variant_id": md.get("variant_id"),
            "product_id": md.get("product_id"),
        }
        # copy only requested keys that exist in metadata
        for key in entities_cap:
            if key in md:
                item[key] = md.get(key)
        matches.append(item)

    # Optional local flexible checks (cheap on <=5 results)
    filtered_matches = [
        item for item in matches
        if all(
            (key in item and flexible_compare(item.get(key), value))
            for key, value in entities_cap.items()
            if value not in [None, "", [], {}]
        )
    ]
    filtered_matches.sort(key=lambda x: (x["score"] is not None, x["score"]), reverse=True)
    return filtered_matches


# --- (Optional) Legacy helper kept for compatibility; no longer used in the fast path ---
async def fetch_records_by_ids(ids: List[str], index_name: str = TEXT_INDEX_NAME, namespace: str = NAMESPACE):
    """
    DEPRECATED: Not needed now that we query with include_metadata=True.
    """
    index = pinecone.Index(index_name)

    def _do_fetch():
        return index.fetch(ids=ids, namespace=namespace)

    response = await anyio.to_thread.run_sync(_do_fetch)
    out = []
    for id_, info in getattr(response, "vectors", {}).items():
        md = getattr(info, "metadata", {}) or {}
        if md:
            out.append(md | {"id": id_})
    return out