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
import unicodedata
from app.core.image_clip import get_image_clip_embedding, get_text_clip_embedding
from app.vector.pinecone_client import get_image_index

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
logger.info("Pinecone index: %s | namespace: %s", TEXT_INDEX_NAME, NAMESPACE or "(empty)")
TEXT_TOP_K = 100  # lower = faster; raise only if needed


# ---------------- helpers ----------------

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


def _title_keep_spaces(s: str) -> str:
    if not s:
        return s
    s = re.sub(r"\s+", " ", str(s)).strip().lower()
    return " ".join(tok.capitalize() for tok in s.split(" "))


def build_filters_from_entities(tenant_id: int, ents: dict) -> dict:
    """Server-side Pinecone metadata filter with category priority"""
    f = {"tenant_id": {"$eq": tenant_id}}
    
    # ALWAYS enforce category - this is non-negotiable
    if ents.get("category"):
        f["category"] = {"$eq": _title_keep_spaces(ents["category"])}

    # Rental preference
    if ents.get("is_rental") is not None:
        f["is_rental"] = {"$eq": bool(ents.get("is_rental"))}

    # Size (critical): coerce numeric-like to number; keep letters with proper casing
    if ents.get("size"):
        sval = ents["size"]
        if isinstance(sval, (int, float)):
            f["size"] = {"$eq": float(sval)}
        elif isinstance(sval, str) and sval.strip().isdigit():
            try:
                f["size"] = {"$eq": float(sval.strip())}
            except Exception:
                f["size"] = {"$eq": smart_capitalize(sval)}
        else:
            # letter sizes like S/M/L/XL/XXL/... -> use smart_capitalize to keep XL, 3XL, etc.
            f["size"] = {"$eq": smart_capitalize(str(sval))}

    if ents.get("type") and ents.get("type_locked"):
        f["type"] = {"$eq": _title_keep_spaces(ents["type"])}

    # Color and fabric (more flexible)
    if ents.get("color"):
        f["color"] = {"$eq": _title_keep_spaces(ents["color"])}
    if ents.get("fabric"):
        f["fabric"] = {"$eq": _title_keep_spaces(ents["fabric"])}

    # Occasion rule unchanged
    if ents.get("occasion") and any(k in f for k in ("color","size","fabric")):
        f["occasion"] = {"$eq": _title_keep_spaces(ents["occasion"])}

    return f




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
    if isinstance(val, list):  # Handle list directly
        return [str(v).strip().capitalize() for v in val if str(v).strip()]
    elif isinstance(val, str):  # Split string on comma
        return [v.strip().capitalize() for v in val.split(",") if v.strip()]
    return []  # Fallback for other types


def _clean_str(x):
    # lower, NFKD, strip accents, remove all non-alphanumerics
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    return re.sub(r"[^a-z0-9]+", "", s)  # "Jimmy Chu" -> "jimmychu"


def flexible_compare(a, b) -> bool:
    """Robust equality for mixed types with space/punct-insensitive string compare."""
    # Booleans
    if isinstance(a, bool) or isinstance(b, bool):
        def to_bool(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.strip().lower() in {"true", "yes", "1"}
            return bool(val)
        return to_bool(a) == to_bool(b)

    # Strings / everything else -> compare normalized forms
    a_clean = _clean_str(a)
    b_clean = _clean_str(b)

    # exact after cleaning
    if a_clean == b_clean:
        return True

    # substring tolerance (helps partials like "silk" vs "soft silk")
    if len(b_clean) <= 5 or b_clean in a_clean or a_clean in b_clean:
        return True

    return False


def build_metadata_filter(tenant_id: int, entities_cap: dict) -> dict:
    """
    (Optional) Generic Pinecone metadata filter.
    Not used in the main flow, but kept for compatibility.
    """
    f = {"tenant_id": {"$eq": tenant_id}}
    for key in ("category", "size", "color", "fabric", "occasion", "is_rental", "type"):
        val = entities_cap.get(key)
        if val not in [None, "", [], {}]:
            if isinstance(val, list):
                normalized_list = [smart_capitalize(v) for v in val]
                f[key] = {"$in": normalized_list}
            elif isinstance(val, str):
                split_val = [smart_capitalize(v.strip()) for v in val.split(",") if v.strip()]
                if len(split_val) > 1:
                    f[key] = {"$in": split_val}
                else:
                    f[key] = {"$eq": smart_capitalize(val)}
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

def strict_color_compare(item_color: str, requested_color: str) -> bool:
    """Strict exact matching for colors - no partial matches allowed"""
    if not item_color or not requested_color:
        return True  # Skip if either is empty
    
    # Normalize both colors for exact comparison
    item_clean = _clean_str(item_color)
    requested_clean = _clean_str(requested_color)
    
    # Only exact match allowed for colors
    return item_clean == requested_clean

# ---------------- main query ----------------

async def pinecone_fetch_records(entities: dict, tenant_id: int) -> List[Dict[str, Any]]:
    """
    Fast product fetch:
      - Normalizes string entities (capitalize) to match your stored metadata style
      - Cached CLIP text embedding
      - Multi-pass Pinecone query (strict -> loose -> tenant-only)
      - Local flexible filter (space-insensitive)
    """
    # Normalize (capitalize strings if your metadata is capitalized)
    def normalize_entities(e: dict) -> dict:
        out = dict(e or {})
        # Apply dynamic capitalization to text metadata fields
        for key in ("category", "color", "fabric", "occasion", "type", "size"):
            if key in out:
                val = out[key]
                if isinstance(val, list):
                    out[key] = [smart_capitalize(item) for item in val if item]
                elif isinstance(val, str):
                    out[key] = smart_capitalize(val)
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

    # Optional debugging prints for per-field combos (not used for querying)
    COLOR_KEY = "color"
    SIZE_KEY = "size"
    FABRIC_KEY = "fabric"

    color_list = get_multi(entities, "color", COLOR_KEY)
    size_list = get_multi(entities, "size", SIZE_KEY)
    fabric_list = get_multi(entities, "fabric", FABRIC_KEY)

    color_list = color_list if color_list else [None]
    size_list = size_list if size_list else [None]
    fabric_list = fabric_list if fabric_list else [None]

    combos = list(product(color_list, size_list, fabric_list))
    if all(combo == (None, None, None) for combo in combos):
        combos = [(None, None, None)]

    for color, size, fabric in combos:
        filter_dict = {"tenant_id": {"$eq": tenant_id}}
        if color:
            filter_dict[COLOR_KEY] = {"$eq": color}
        if size:
            filter_dict[SIZE_KEY] = {"$eq": size}
        if fabric:
            filter_dict[FABRIC_KEY] = {"$eq": fabric}
        print(f"Sub-query filter: {filter_dict}")

    # Get (cached) embedding as list[float]
    query_vector = embed_text_cached(query_text)

    # 1) strict filter (may include fabric)
    md_filter = build_filters_from_entities(tenant_id, entities_cap)
    logger.info("Pinecone strict filter: %s", md_filter)

    def _do_query(filt: dict, top_k: int):
        return PC_INDEX.query(
            vector=query_vector,
            top_k=top_k,
            namespace=NAMESPACE,
            filter=filt,
            include_metadata=True,
            include_values=False,
        )

    # First try: strict
    
    critical_attributes = ['color', 'size']  # Add other critical attributes
    has_critical_attrs = any(entities_cap.get(attr) for attr in critical_attributes)
    # If nothing came back, retry without fabric (let local fuzzy check handle it)
    response = await anyio.to_thread.run_sync(lambda: _do_query(md_filter, TEXT_TOP_K))
    if not getattr(response, "matches", None) and has_critical_attrs:
        logger.info("No matches found for critical attributes (color/size/fabric/occasion), returning empty")
        return []
    if not getattr(response, "matches", None) and not has_critical_attrs :
        fallback_filter = {"tenant_id": {"$eq": tenant_id}}
        
        # Keep category and is_rental as they're most important
        if entities_cap.get("category"):
            fallback_filter["category"] = {"$eq": entities_cap["category"]}
        if entities_cap.get("is_rental") is not None:
            fallback_filter["is_rental"] = {"$eq": entities_cap["is_rental"]}
        if entities_cap.get("size"):
            fallback_filter["size"] = {"$eq": entities_cap["size"]}
        
        logger.info("Pinecone category-priority fallback: %s", fallback_filter)
        response = await anyio.to_thread.run_sync(lambda: _do_query(fallback_filter, max(TEXT_TOP_K, 50)))

    # If still nothing, try tenant-only (diagnostic safety net)
    if not getattr(response, "matches", None):
        tenant_only = {"tenant_id": {"$eq": tenant_id}}
        logger.warning("Pinecone tenant-only fallback filter: %s", tenant_only)
        response = await anyio.to_thread.run_sync(lambda: _do_query(tenant_only, max(TEXT_TOP_K, 100)))

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
            "price": md.get("price"),
            "rental_price": md.get("rental_price"),
            "variant_id": md.get("variant_id"),
            "product_id": md.get("product_id"),
            "image_url": md.get("image_url"),
            "product_url": md.get("product_url"),
            "type": md.get("type"),
        }
        # copy only requested keys that exist in metadata
        for key in entities_cap:
            if key in md:
                item[key] = md.get(key)
        matches.append(item)

    # Quick preview logging
    num = len(matches)
    logger.info("Pinecone returned matches: %s (index=%s, ns=%s)", num, TEXT_INDEX_NAME, NAMESPACE or "(empty)")
    if num:
        preview = [{"name": m.get("name"), "category": m.get("category"), "fabric": m.get("fabric")} for m in matches[:3]]
        logger.info("Preview (first 3): %s", preview)

    # Optional local flexible checks (cheap on <=5 results)
    def strict_category_match(item, requested_category):
        if not requested_category:
            return True
        item_category = str(item.get("category") or "").strip().lower()
        req_category = str(requested_category or "").strip().lower()
        return item_category == req_category

    # Filter matches with category priority
    SCORE_THRESHOLD = 0.10
    filtered_matches = []

    for item in matches:
        # Must pass score threshold
        if not (item.get("score") is not None and item["score"] >= SCORE_THRESHOLD):
            continue
        
        # STRICT: Must match category if specified
        if entities_cap.get("category") and not strict_category_match(item, entities_cap.get("category")):
            continue
        
        # STRICT COLOR MATCHING - this is the key fix
        if entities_cap.get("color"):
            if not strict_color_compare(item.get("color"), entities_cap.get("color")):
                logger.info("Color mismatch: item=%s vs requested=%s", item.get("color"), entities_cap.get("color"))
                continue
        
        # Flexible matching for other attributes
        match = True
        for key, value in entities_cap.items():
            if key in ["category", "color"]:  # Already handled above
                continue
            if key == "occasion" and not item.get("occasion"):
                continue  # Skip occasion check if item has no occasion
            if value not in [None, "", [], {}] and key in item:
                if not flexible_compare(item.get(key), value):
                    match = False
                    break
        
        if match:
            filtered_matches.append(item)

    # filtered_matches = [
    #     item for item in matches
    #     if (item.get("score") is not None and item["score"] >= SCORE_THRESHOLD) and
    #     all(
    #         (key in item and flexible_compare(item.get(key), value))
    #         for key, value in entities_cap.items()
    #         if value not in [None, "", [], {}]
    #     )
    # ]
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

async def search_products_by_image(image_url, top_k=5, namespace=NAMESPACE):
    img_emb = get_image_clip_embedding(image_url)  # This will be 512-dim if using ViT-B-32
    index = get_image_index()
    print('index :', index.describe_index_stats())
    res = index.query(
        vector=img_emb,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    print('res..................122522', res)
    products = []
    for match in res.get("matches", []):
        meta = match.get("metadata", {})
        products.append({
            "id": meta.get("id", "MISSING"),
            "product_name": meta.get("product_name", "MISSING"),
            "image_url": meta.get("image_url", "MISSING"),
            "score": match.get("score", 0)
        })
    return products

