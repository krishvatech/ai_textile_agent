import os
import asyncio
import logging
from functools import lru_cache
from typing import Dict, Any, List
import anyio
import re
from itertools import product
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
TEXT_TOP_K = 50  # lower = faster; raise only if needed

def smart_capitalize(text: str) -> str:
    """Dynamic title casing:
       - Capitalize each word
       - Keep size tokens (xl, xxl, 3xl, etc.) fully upper
       - Handle hyphen/ slash/ underscore separated parts
    """
    if not isinstance(text, str):
        return text
    s = re.sub(r"\s+", " ", text.strip())
    if not s:
        return s

    SIZE_PAT = re.compile(r"^(?:\d+)?(?:(?:xxs|xs|s|m|l|xl|xxl|xxxl)|\d+xl)$", re.I)

    def cap_token(tok: str) -> str:
        # preserve all-uppercase tokens that look like sizes (xl/xxl/3xl etc.)
        if SIZE_PAT.match(tok):
            return tok.upper()

        # recurse for compound tokens
        for sep in ("-", "/", "_"):
            if sep in tok:
                return sep.join(cap_token(part) for part in tok.split(sep) if part != "")

        # normal word -> first letter upper, rest lower
        return tok[:1].upper() + tok[1:].lower() if tok else tok

    return " ".join(cap_token(t) for t in s.split(" "))

def normalize_category(cat: str) -> str:
    # For now, category uses the same dynamic logic
    return smart_capitalize(cat)

def get_multi(entities, field, actual_key=None):
    """
    entities: user query dict
    field: key to grab from user input
    actual_key: metadata key in Pinecone (case-sensitive!)
    """
    val = entities.get(field)
    if val is None:
        val = entities.get(field.capitalize())
    if val is None and actual_key is not None:
        val = entities.get(actual_key)
    if isinstance(val, list):  # NEW: Handle list directly
        return [str(v).strip().capitalize() for v in val if str(v).strip()]
    elif isinstance(val, str):  # Existing: Split string on comma
        return [v.strip().capitalize() for v in val.split(",") if v.strip()]
    return []  # Fallback for other types

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
            if isinstance(val, list):
                normalized_list = [smart_capitalize(v) for v in val]  # NEW: Capitalize list items
                f[key] = {"$in": normalized_list}
            elif isinstance(val, str):
                split_val = [smart_capitalize(v.strip()) for v in val.split(",") if v.strip()]  # NEW: Split and capitalize
                if len(split_val) > 1:
                    f[key] = {"$in": split_val}
                else:
                    f[key] = {"$eq": val}
            elif isinstance(val, bool):
                f[key] = {"$eq": bool(val)}
            else:
                f[key] = {"$eq": val}
    return f


def flexible_compare(a, b) -> bool:
    """Robust equality for mixed types (bool/str/num). Add fuzzy matching for strings."""
    if isinstance(a, bool) or isinstance(b, bool):
        def to_bool(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.strip().lower() in {"true", "yes", "1"}
            return bool(val)
        return to_bool(a) == to_bool(b)
    
    a_str = str(a).lower().strip()
    b_str = str(b).lower().strip()
    
    # Strict equality for most fields
    if a_str == b_str:
        return True
    
    # Fuzzy for specific fields (e.g., fabric can be partial)
    # Allow substring matches if the query is shorter
    if len(b_str) <= 5 or b_str in a_str or a_str in b_str:  # e.g., 'silk' in 'soft silk'
        return True
    
    return False


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
    def normalize_entities(e: dict) -> dict:
        out = dict(e or {})
        # Apply dynamic capitalization to text metadata fields
        for key in ("category", "color", "fabric", "occasion", "type", "size"):
            if key in out:
                val = out[key]
                if isinstance(val, list):  # NEW: Handle lists by capitalizing each item
                    out[key] = [smart_capitalize(item) for item in val if item]
                elif isinstance(val, str):
                    out[key] = smart_capitalize(val)
                # Else, leave as is
        # Optional: coerce is_rental to real bool if it came as string
        if "is_rental" in out and isinstance(out["is_rental"], str):
            out["is_rental"] = out["is_rental"].strip().lower() in {"true", "1", "yes", "y"}
        # Remove empties
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}

    
    entities_cap = normalize_entities(entities)
    logger.info("normalized entities: %s", entities_cap)
    print(entities_cap)
    # Build compact text query from non-empty strings
    texts = [str(v) for v in entities_cap.values() if isinstance(v, str) and v.strip()]
    if not texts:
        logger.info("pinecone_fetch_records: no non-empty string fields in entities; returning [].")
        return []

    query_text = " ".join(texts)
    text_input = tokenizer([query_text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
    COLOR_KEY   = "color"   # For color field—adjust if your index uses 'Color'
    SIZE_KEY    = "size"    # == 'size' (small-case, as per your confirmation!)
    FABRIC_KEY  = "fabric"  # For fabric field—adjust if needed

    color_list  = get_multi(entities, "color", COLOR_KEY)
    size_list   = get_multi(entities, "size", SIZE_KEY)       # Now using 'size'
    fabric_list = get_multi(entities, "fabric", FABRIC_KEY)

    color_list  = color_list  if color_list  else [None]
    size_list   = size_list   if size_list   else [None]
    fabric_list = fabric_list if fabric_list else [None]

    
    combos = list(product(color_list, size_list, fabric_list))
    if all(combo == (None, None, None) for combo in combos):
        combos = [(None, None, None)]

    for color, size, fabric in combos:
        filter_dict = {"tenant_id": {"$eq": tenant_id}}
        if color:
            filter_dict[COLOR_KEY] = {"$eq": color}
        if size:
            filter_dict[SIZE_KEY] = {"$eq": size}          # small-case 'size'
        if fabric:
            filter_dict[FABRIC_KEY] = {"$eq": fabric}
        print(f"Sub-query filter: {filter_dict}")
        
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
    print(response)
    # Build matches directly from returned metadata (single round-trip)
    matches: List[Dict[str, Any]] = []
    for m in getattr(response, "matches", []) or []:
        md = getattr(m, "metadata", {}) or {}
        item = {
            "id": getattr(m, "id", None),
            "score": getattr(m, "score", None),
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
        }
        # copy only requested keys that exist in metadata
        for key in entities_cap:
            if key in md:
                item[key] = md.get(key)
        matches.append(item)
    SCORE_THRESHOLD = 0.35
    # Optional local flexible checks (cheap on <=5 results)
    filtered_matches = [
        item for item in matches
        if (item.get("score") is not None and item["score"] >= SCORE_THRESHOLD) and
        all(
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

async def demo():
    tenant_id = 4  # <-- change if needed
    entities = {
        "category": "saree",
    }

    results = await pinecone_fetch_records(entities=entities, tenant_id=tenant_id)

    if not results:
        print("No matches")
        return

    for i, it in enumerate(results, 1):
        print(f"\n#{i}")
        print(" id:", it.get("id"))
        print(" score:", it.get("score"))
        print(" product_name:", it.get("product_name"))
        print(" is_rental:", it.get("is_rental"))
        print(" category:", it.get("category"))
        print(" occasion:", it.get("occasion"))
        print(" fabric:", it.get("fabric"))
        print(" color:", it.get("color"))
        print(" size:", it.get("size"))
        print(" variant_id:", it.get("variant_id"))
        print(" product_id:", it.get("product_id"))

if __name__ == "__main__":
    asyncio.run(demo())