# #!/usr/bin/env python3
# # add_product.py — Pinecone-only upsert (text + image) into ONE index with auto-create
# # - Index: ai-textile-agent (or PINECONE_INDEX env)
# # - If index missing, creates serverless index (dim=512, cosine)
# # - model_name: ViT-B-32-quickgelu (image tower); CLIP text tower matches that space

# import os
# import json
# from io import BytesIO
# from hashlib import md5
# from typing import Any, Dict, List, Optional
# from dotenv import load_dotenv
# import requests
# from PIL import Image

# import torch
# import open_clip
# from sentence_transformers import SentenceTransformer
# from pinecone import Pinecone, ServerlessSpec

# # ================= ENV =================
# load_dotenv()

# PINECONE_API_KEY = "pcsk_27uP34_94NTBKhAoJkJ7TjtJSw2isBjURag5vfHBTUoMANfrV6pQ5TXV1Dv68NkEhSrhxn"
# PINECONE_INDEX = "ai-textile-agent"
# PINECONE_NAMESPACE = "default"
# PINECONE_CLOUD = "aws"
# PINECONE_REGION = "us-east-1"

# MODEL_NAME_IMAGE = "ViT-B-32-quickgelu"  # requested
# MODEL_NAME_TEXT = "clip-ViT-B-32"        # CLIP text tower aligned with ViT-B/32

# if not PINECONE_API_KEY:
#     raise RuntimeError("PINECONE_API_KEY not set")

# # ================= CLIENTS / MODELS =================
# pc = Pinecone(api_key=PINECONE_API_KEY)

# def ensure_index():
#     """Create serverless index if it doesn't exist (512-dim, cosine)."""
#     try:
#         # try to list by names (compat across client versions)
#         names = []
#         try:
#             names = getattr(pc.list_indexes(), "names", lambda: [])()
#         except Exception:
#             listed = pc.list_indexes()
#             if isinstance(listed, (list, tuple)):
#                 names = [getattr(x, "name", getattr(x, "get", lambda k, d=None: None)("name")) for x in listed]
#             elif isinstance(listed, dict) and "indexes" in listed:
#                 names = [idx.get("name") for idx in listed["indexes"]]
#         if PINECONE_INDEX not in set(filter(None, names)):
#             print(f"Creating index '{PINECONE_INDEX}' (dim=512, cosine, {PINECONE_CLOUD}/{PINECONE_REGION}) ...")
#             pc.create_index(
#                 name=PINECONE_INDEX,
#                 dimension=512,
#                 metric="cosine",
#                 spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
#             )
#     except Exception as e:
#         try:
#             pc.describe_index(PINECONE_INDEX)  # will raise if missing
#         except Exception:
#             raise RuntimeError(f"Failed to ensure index '{PINECONE_INDEX}': {e}")

# ensure_index()
# pc_index = pc.Index(PINECONE_INDEX)

# # Text tower (512-dim CLIP text)
# st_model = SentenceTransformer(MODEL_NAME_TEXT)

# # Image tower (512-dim CLIP image)
# _device = "cpu"
# try:
#     torch.set_num_threads(min(4, (os.cpu_count() or 4)))
# except Exception:
#     pass

# clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
#     MODEL_NAME_IMAGE, pretrained="openai"
# )
# clip_model.eval().to(_device)

# # ================= HELPERS =================
# def to_float(v):
#     try:
#         return float(v)
#     except (TypeError, ValueError):
#         return None

# def as_bool(v):
#     if isinstance(v, bool):
#         return v
#     if v is None:
#         return False
#     s = str(v).strip().lower()
#     return s in {"1", "true", "yes", "y", "on"}

# def text_embed(s: str):
#     s = (s or "").strip()
#     vec = st_model.encode(s, normalize_embeddings=True)
#     return vec.tolist()  # 512 floats

# def image_embed_from_url(url: str):
#     if not url or not isinstance(url, str):
#         raise ValueError("empty image url")
#     resp = requests.get(url.strip(), timeout=25)
#     resp.raise_for_status()
#     img = Image.open(BytesIO(resp.content)).convert("RGB")
#     with torch.inference_mode():
#         x = clip_preprocess(img).unsqueeze(0).to(_device)
#         feats = clip_model.encode_image(x)
#         feats = feats / feats.norm(dim=-1, keepdim=True)
#     return feats[0].cpu().numpy().astype("float32").tolist()  # 512 floats

# def stable_int_id(*parts) -> int:
#     """Stable integer from a string hash (for missing product/variant ids)."""
#     base = "§".join([str(p) for p in parts if p is not None])
#     return int(md5(base.encode("utf-8")).hexdigest()[:12], 16)

# def build_text_string(vd: dict) -> str:
#     return " ".join([
#         vd.get("name", "") or "",
#         vd.get("color", "") or "",
#         vd.get("fabric", "") or "",
#         vd.get("category", "") or "",
#         vd.get("type", "") or "",
#         vd.get("occasion", "") or "",
#         vd.get("description", "") or "",
#     ]).strip()

# def extract_image_urls(obj: Dict[str, Any]) -> List[str]:
#     """
#     Extract image URLs from common fields/shapes:
#       - "image_url": str or list[str]
#       - "image", "img", "primary_image": str
#       - "images", "image_urls", "imageUrls", "gallery": list[str|dict]
#          dict item keys checked: 'url', 'src', 'image', 'image_url', 'link'
#     """
#     urls: List[str] = []

#     # direct single fields
#     for k in ["image_url", "image", "img", "primary_image", "imageUrl"]:
#         v = obj.get(k)
#         if isinstance(v, str) and v.strip():
#             urls.append(v.strip())
#         elif isinstance(v, list):
#             urls.extend([x.strip() for x in v if isinstance(x, str) and x.strip()])

#     # plural/collection fields
#     for k in ["images", "image_urls", "imageUrls", "gallery", "photos"]:
#         v = obj.get(k)
#         if isinstance(v, list):
#             for item in v:
#                 if isinstance(item, str) and item.strip():
#                     urls.append(item.strip())
#                 elif isinstance(item, dict):
#                     for kk in ["url", "src", "image", "image_url", "link"]:
#                         vv = item.get(kk)
#                         if isinstance(vv, str) and vv.strip():
#                             urls.append(vv.strip())

#     # de-dupe preserve order
#     seen = set()
#     out = []
#     for u in urls:
#         if u not in seen:
#             seen.add(u)
#             out.append(u)
#     return out

# def upsert_text_and_images(variant_data: dict):
#     """
#     Upserts into ONE index / namespace:
#       - TEXT vector id: txt:{tenant}:{product}:{variant}
#       - IMAGE vector id(s): img:{tenant}:{product}:{variant}:{idx}:{hash8}
#     Metadata includes modality and model_name="ViT-B-32-quickgelu".
#     """
#     tenant_id = variant_data.get("tenant_id")
#     product_id = variant_data.get("product_id")
#     variant_id = variant_data.get("variant_id")

#     # Generate stable ids if missing
#     if product_id is None:
#         product_id = stable_int_id(tenant_id, variant_data.get("name"), variant_data.get("category"))
#     if variant_id is None:
#         variant_id = stable_int_id(product_id, variant_data.get("color"), variant_data.get("size"),
#                                    variant_data.get("image_url"))

#     # ---------- TEXT VECTOR ----------
#     txt = build_text_string(variant_data)
#     txt_vec = text_embed(txt)

#     base_meta = {
#         "tenant_id": int(tenant_id) if tenant_id is not None else None,
#         "product_id": int(product_id),
#         "variant_id": int(variant_id),
#         "name": variant_data.get("name") or "",
#         "category": variant_data.get("category") or "",
#         "type": variant_data.get("type") or "",
#         "fabric": variant_data.get("fabric") or "",
#         "color": variant_data.get("color") or "",
#         "size": variant_data.get("size") or "",
#         "occasion": variant_data.get("occasion") or "",
#         "price": to_float(variant_data.get("price")),
#         "rental_price": to_float(variant_data.get("rental_price")),
#         "available_stock": int(variant_data.get("available_stock") or 0),
#         "product_url": variant_data.get("product_url") or "",
#         "is_rental": as_bool(variant_data.get("is_rental")),
#         "is_active": as_bool(variant_data.get("is_active", True)),
#         "description": variant_data.get("description") or "",
#         "model_name": MODEL_NAME_IMAGE,  # requested
#     }
#     text_meta = {
#         **{k: v for k, v in base_meta.items() if v is not None},
#         "modality": "text",
#         "image_url": variant_data.get("image_url") or "",
#     }

#     text_vec_id = f"txt:{tenant_id}:{product_id}:{variant_id}"
#     pc_index.upsert(
#         vectors=[{"id": text_vec_id, "values": txt_vec, "metadata": text_meta}],
#         namespace=PINECONE_NAMESPACE
#     )

#     # ---------- IMAGE VECTOR(S) ----------
#     urls = []
#     # allow comma-separated string or list already present
#     raw = variant_data.get("image_url")
#     if isinstance(raw, str) and raw.strip():
#         urls = [u.strip() for u in raw.split(",") if u.strip()]
#     elif isinstance(raw, list):
#         urls = [u.strip() for u in raw if isinstance(u, str) and u.strip()]

#     # if still empty, try to discover from the whole record
#     if not urls:
#         urls = extract_image_urls(variant_data)

#     if urls:
#         for idx, url in enumerate(urls):
#             try:
#                 img_vec = image_embed_from_url(url)
#             except Exception as e:
#                 print(f"[warn] image embed failed for variant {variant_id} url={url}: {e}")
#                 continue

#             suffix = md5(url.encode("utf-8")).hexdigest()[:8]
#             img_vec_id = f"img:{tenant_id}:{product_id}:{variant_id}:{idx}:{suffix}"

#             img_meta = {
#                 **{k: v for k, v in base_meta.items() if v is not None},
#                 "modality": "image",
#                 "image_url": url,  # this specific image
#             }

#             pc_index.upsert(
#                 vectors=[{"id": img_vec_id, "values": img_vec, "metadata": img_meta}],
#                 namespace=PINECONE_NAMESPACE
#             )

#     print(f"Upserted TEXT+IMAGE for variant {variant_id} into index '{PINECONE_INDEX}'.")

# # ================= BATCH LOADER =================
# def _normalize_occasion(val: Any) -> str:
#     if isinstance(val, list):
#         return " ".join([x for x in val if x])
#     return (val or "").strip()

# def _synthesize_variant_from_product(rec: Dict[str, Any]) -> Dict[str, Any]:
#     """Build a single variant dict using top-level product fields."""
#     urls = extract_image_urls(rec)
#     return {
#         "tenant_id": rec.get("tenant_id"),
#         "product_id": rec.get("product_id"),
#         "variant_id": rec.get("variant_id") or rec.get("id") or rec.get("sku"),
#         "name": rec.get("product_name") or rec.get("name") or "",
#         "category": rec.get("category") or "",
#         "type": rec.get("type") or rec.get("gender") or "",
#         "description": rec.get("description") or "",
#         "product_url": rec.get("product_url") or "",
#         "color": rec.get("color") or "",
#         "size": rec.get("size") or rec.get("default_size") or "",
#         "fabric": rec.get("fabric") or "",
#         "price": rec.get("price"),
#         "rental_price": rec.get("rental_price"),
#         "available_stock": rec.get("available_stock", 0),
#         "image_url": ", ".join(urls) if urls else rec.get("image_url") or "",
#         "is_rental": rec.get("is_rental", False),
#         "is_active": rec.get("is_active", True),
#         "occasion": _normalize_occasion(rec.get("occasion")),
#     }

# def upsert_from_file(file_path: str):
#     """
#     Expects a JSON array. Works whether each record has `variants` or not.
#     """
#     with open(file_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     if not data or not isinstance(data, list):
#         print("No valid records found (expecting JSON array).")
#         return

#     for rec in data:
#         # Common product-level fields
#         product_name = rec.get("product_name") or rec.get("name") or ""

#         variants = rec.get("variants")
#         if isinstance(variants, list) and variants:
#             # A) Product with variants
#             tenant_id = rec.get("tenant_id")
#             product_id = rec.get("product_id")
#             category = rec.get("category") or ""
#             type_ = rec.get("type") or rec.get("gender") or ""
#             description = rec.get("description") or ""
#             product_url = rec.get("product_url") or ""

#             for v in variants:
#                 vd = {
#                     "tenant_id": tenant_id,
#                     "product_id": product_id,           # can be None
#                     "variant_id": v.get("variant_id"),  # can be None
#                     "name": product_name,
#                     "category": category,
#                     "type": type_,
#                     "description": description,
#                     "product_url": v.get("product_url") or product_url,
#                     "color": v.get("color") or "",
#                     "size": v.get("size") or "",
#                     "fabric": v.get("fabric") or "",
#                     "price": v.get("price"),
#                     "rental_price": v.get("rental_price"),
#                     "available_stock": v.get("available_stock", 0),
#                     "image_url": v.get("image_url") or rec.get("image_url") or "",
#                     "is_rental": v.get("is_rental", False),
#                     "is_active": v.get("is_active", True),
#                     "occasion": _normalize_occasion(v.get("occasion")),
#                 }
#                 upsert_text_and_images(vd)
#         else:
#             # B) Product without variants — synthesize a single variant
#             vd = _synthesize_variant_from_product(rec)
#             if vd.get("name") or vd.get("image_url"):
#                 upsert_text_and_images(vd)
#             else:
#                 print(f"[skip] Not enough data to upsert product '{product_name}'")

#     print("Done: all valid records upserted into Pinecone.")

# # ================= ENTRY =================
# if __name__ == "__main__":
#     # Replace with your file path, e.g. "app/tests/your_products.json"
#     upsert_from_file("app/tests/sarees_casual_top10_modified_json.json")














#!/usr/bin/env python3
"""
add_product_images_only.py — Insert products/variants into Postgres and upsert
ONLY IMAGE embeddings to Pinecone for visual search.

• Image tower: open_clip ViT-B-32-quickgelu (512-dim CLIP image space)
• Vector IDs: img:{tenant_id}:{product_id}:{variant_id}:{idx}:{md5-of-url-8}
• Metadata: tenant_id, product_id, variant_id, name, category, type, fabric,
  color, size, occasion, price, rental_price, available_stock, product_url,
  is_rental, is_active, description, model_name, modality=image, image_url
• Reads same JSON array you use for add_product, inserts into Postgres, then
  pushes one vector per image URL for each variant to Pinecone.

ENV required:
  PINECONE_API_KEY
  PINECONE_INDEX (default: textile-products)
  PINECONE_NAMESPACE (default: default)
  PINECONE_CLOUD (default: aws)
  PINECONE_REGION (default: us-east-1)

Run:
  python add_product_images_only.py path/to/variants.json
"""

from __future__ import annotations

import os
import sys
import json
import enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from io import BytesIO
from hashlib import md5

import requests
from PIL import Image

import torch
import open_clip
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from app.db.db_connection import get_db_connection, close_db_connection

# ================== ENV ==================
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "textile-products")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set")

# ================== MODEL ==================
try:
    torch.set_num_threads(min(4, (os.cpu_count() or 4)))
except Exception:
    pass

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu", pretrained="openai"
)
clip_model.eval()

# ================== PINECONE ==================
pc = Pinecone(api_key=PINECONE_API_KEY)

def ensure_index() -> None:
    try:
        # Handle both SDK shapes of list_indexes
        names = []
        try:
            names = getattr(pc.list_indexes(), "names", lambda: [])()
        except Exception:
            listed = pc.list_indexes()
            if isinstance(listed, (list, tuple)):
                names = [getattr(x, "name", None) or (x.get("name") if isinstance(x, dict) else None) for x in listed]
            elif isinstance(listed, dict) and "indexes" in listed:
                names = [idx.get("name") for idx in listed["indexes"]]
        if PINECONE_INDEX not in {n for n in names if n}:
            print(f"Creating index '{PINECONE_INDEX}' (dim=512, cosine, {PINECONE_CLOUD}/{PINECONE_REGION}) ...")
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=512,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
    except Exception as e:
        try:
            pc.describe_index(PINECONE_INDEX)
        except Exception:
            raise RuntimeError(f"Failed to ensure index '{PINECONE_INDEX}': {e}")

ensure_index()
pinecone_index = pc.Index(PINECONE_INDEX)

# ================== ENUMS ==================
class RentalStatus(str, enum.Enum):
    active = "active"
    returned = "returned"
    cancelled = "cancelled"

# ================== HELPERS ==================

def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def extract_image_urls(obj: Dict[str, Any]) -> List[str]:
    """Collect possible image URLs from common shapes/fields.
    Supports: image_url (str|list), image/img/primary_image/imageUrl,
    and lists under images/image_urls/imageUrls/gallery/photos with optional dict items.
    """
    urls: List[str] = []

    # direct single fields
    for k in ["image_url", "image", "img", "primary_image", "imageUrl"]:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            urls.append(v.strip())
        elif isinstance(v, list):
            urls.extend([x.strip() for x in v if isinstance(x, str) and x.strip()])

    # plural/collection fields
    for k in ["images", "image_urls", "imageUrls", "gallery", "photos"]:
        v = obj.get(k)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str) and item.strip():
                    urls.append(item.strip())
                elif isinstance(item, dict):
                    for kk in ["url", "src", "image", "image_url", "link"]:
                        vv = item.get(kk)
                        if isinstance(vv, str) and vv.strip():
                            urls.append(vv.strip())

    # de-dupe preserve order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _download_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


@torch.inference_mode()
def image_embed_from_url(url: str) -> List[float]:
    img = _download_image(url)
    img_t = clip_preprocess(img).unsqueeze(0)
    feats = clip_model.encode_image(img_t)
    feats = feats.float()
    feats /= feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().tolist()  # 512 floats


def _common_metadata(variant_data: dict) -> dict:
    return {
        "tenant_id": int(variant_data["tenant_id"]) if variant_data.get("tenant_id") is not None else None,
        "product_id": int(variant_data["product_id"]),
        "variant_id": int(variant_data["variant_id"]),
        "name": variant_data.get("name") or "",
        "category": variant_data.get("category") or "",
        "type": variant_data.get("type") or "",
        "fabric": variant_data.get("fabric") or "",
        "color": variant_data.get("color") or "",
        "size": variant_data.get("size") or "",
        "occasion": variant_data.get("occasion") or "",
        "price": _to_float(variant_data.get("price")),
        "rental_price": _to_float(variant_data.get("rental_price")),
        "available_stock": int(variant_data.get("available_stock") or 0),
        "product_url": variant_data.get("product_url") or "",
        "is_rental": _as_bool(variant_data.get("is_rental")),
        "is_active": _as_bool(variant_data.get("is_active", True)),
        "description": variant_data.get("description") or "",
        "model_name": "ViT-B-32-quickgelu",
    }


def upsert_images_only(variant_data: dict):
    tenant_id = variant_data.get("tenant_id")
    product_id = variant_data.get("product_id")
    variant_id = variant_data.get("variant_id")

    # Gather URLs from various shapes
    urls: List[str] = []
    raw = variant_data.get("image_url")
    if isinstance(raw, str) and raw.strip():
        # support comma-separated strings
        urls = [u.strip() for u in raw.split(",") if u.strip()]
    elif isinstance(raw, list):
        urls = [u.strip() for u in raw if isinstance(u, str) and u.strip()]

    if not urls:
        urls = extract_image_urls(variant_data)

    if not urls:
        print(f"[img-skip] variant {variant_id}: no image URL found")
        return

    for idx, url in enumerate(urls):
        try:
            img_vec = image_embed_from_url(url)
        except Exception as e:
            print(f"[warn] image embed failed for variant {variant_id} url={url}: {e}")
            continue

        suffix = md5(url.encode("utf-8")).hexdigest()[:8]
        img_vec_id = f"img:{tenant_id}:{product_id}:{variant_id}:{idx}:{suffix}"

        img_meta = {
            **{k: v for k, v in _common_metadata(variant_data).items() if v is not None},
            "modality": "image",
            "image_url": url,  # specific image URL
        }

        pinecone_index.upsert(
            vectors=[{"id": img_vec_id, "values": img_vec, "metadata": img_meta}],
            namespace=PINECONE_NAMESPACE,
        )

    print(f"Upserted IMAGE vectors for variant {variant_id} into index '{PINECONE_INDEX}'.")


# ================== OCCASION HELPER ==================

def get_or_create_occasion_id(cursor, occasion_name: Optional[str]):
    if not occasion_name or not occasion_name.strip():
        return None
    cursor.execute("SELECT id FROM occasions WHERE name = %s", (occasion_name.strip(),))
    result = cursor.fetchone()
    if result:
        return result[0]
    cursor.execute(
        "INSERT INTO occasions (name) VALUES (%s) RETURNING id",
        (occasion_name.strip(),)
    )
    occasion_id = cursor.fetchone()[0]
    print(f"Inserted new occasion '{occasion_name}' with ID {occasion_id}")
    return occasion_id


# ================== MAIN BATCH LOGIC ==================

def auto_batch_insert_from_file(file_path: str):
    conn, cursor = get_db_connection()
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Could not parse JSON: {e}")
            close_db_connection(conn, cursor)
            return

    if not data or not isinstance(data, list):
        print("No valid records found in file (expecting JSON array). Exiting.")
        close_db_connection(conn, cursor)
        return

    # Group unique products
    products: Dict[Any, Dict[str, Any]] = {}
    for variant in data:
        pid = variant.get('product_id')
        if pid and pid not in products:
            products[pid] = {
                'tenant_id': variant.get('tenant_id'),
                'name': variant.get('product_name', ''),
                'category': variant.get('category', ''),
                'description': variant.get('description', ''),
                'type': variant.get('type', ''),
            }

    try:
        timestamp = datetime.now()
        product_id_map: Dict[Any, int] = {}

        # Insert products
        for old_pid, prod in products.items():
            if not all([prod['tenant_id'], prod['name'], prod['category'], prod['type']]):
                print(f"Skipping product {old_pid} due to missing fields.")
                continue
            cursor.execute(
                '''
                INSERT INTO products (tenant_id, name, category, description, type, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
                ''',
                (prod['tenant_id'], prod['name'], prod['category'], prod['description'], prod['type'],
                 timestamp, timestamp)
            )
            new_pid = cursor.fetchone()[0]
            product_id_map[old_pid] = new_pid
            print(f"Inserted product '{prod['name']}' with new ID: {new_pid}")

        # Insert variants
        for variant in data:
            old_pid = variant.get('product_id')
            if old_pid not in product_id_map:
                print(f"Skipping variant due to missing product ID: {old_pid}")
                continue

            new_pid = product_id_map[old_pid]
            price = variant.get('price')
            rental_price = variant.get('rental_price')
            available_stock = variant.get('available_stock', 0)
            if price is None:
                print(f"Skipping variant due to missing price: {variant.get('product_name')}")
                continue

            is_active = not variant.get('extra', {}).get('is_demo', False)
            is_rental_raw = variant.get('is_rental', False)
            if isinstance(is_rental_raw, str):
                is_rental = is_rental_raw.lower() in ('true', '1', 'yes')
            else:
                is_rental = bool(is_rental_raw)

            image_url = variant.get('image_url')
            product_url = variant.get('product_url')

            # Occasion normalize
            occasions_raw = variant.get('occasion', [])
            if isinstance(occasions_raw, list):
                occasions = [o for o in occasions_raw if o]
            elif isinstance(occasions_raw, str) and occasions_raw.strip():
                occasions = [occasions_raw.strip()]
            else:
                occasions = []
            print(f"Processing occasions for variant '{variant.get('product_name')}': {occasions}")
            occasion_str = ' '.join(occasions) if occasions else ''

            try:
                price_float = float(price)
                rental_price_float = float(rental_price) if rental_price not in (None, '', 'null') else None
                stock_int = int(available_stock) if available_stock is not None else 0
            except (ValueError, TypeError) as conv_err:
                print(f"Conversion error for variant {variant.get('product_name')}: {conv_err}")
                continue

            cursor.execute(
                '''
                INSERT INTO product_variants (
                    product_id, color, size, fabric, price, available_stock,
                    is_rental, rental_price, image_url, created_at, updated_at, product_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                ''',
                (
                    new_pid, variant.get('color'), variant.get('size'), variant.get('fabric'),
                    price_float, stock_int, is_rental, rental_price_float, image_url,
                    timestamp, timestamp, product_url
                )
            )
            variant_id = cursor.fetchone()[0]
            print(f"Inserted variant '{variant.get('product_name')}' ({variant.get('color')}, {variant.get('size')}) with ID: {variant_id}")

            # Link occasions
            for occ in occasions:
                if not occ:
                    continue
                occ_id = get_or_create_occasion_id(cursor, occ)
                if not occ_id:
                    print(f"Skipped blank/invalid occasion for: {occ}")
                    continue
                cursor.execute(
                    '''INSERT INTO product_variant_occasions (variant_id, occasion_id) VALUES (%s, %s)''',
                    (variant_id, occ_id)
                )
                print(f"Linked variant {variant_id} to occasion '{occ}' (ID {occ_id})")

            # -------- Pinecone: IMAGE upsert ONLY --------
            variant_data = {
                "variant_id": variant_id,
                "product_id": new_pid,
                "tenant_id": variant.get('tenant_id'),
                "name": variant.get('product_name', ''),
                "color": variant.get('color', ''),
                "size": variant.get('size', ''),
                "fabric": variant.get('fabric', ''),
                "category": variant.get('category', ''),
                "type": variant.get('type', ''),
                "occasion": occasion_str,
                "price": price_float,
                "rental_price": rental_price_float,
                "available_stock": stock_int,
                "image_url": image_url if image_url is not None else "",
                "product_url": product_url if product_url is not None else "",
                "is_rental": is_rental,
                "is_active": is_active,
                "description": variant.get('description', ''),
            }
            upsert_images_only(variant_data)

        conn.commit()
        print("All valid records inserted into Postgres and IMAGE embeddings upserted to Pinecone.")
    except Exception as e:
        conn.rollback()
        print(f"Error during auto batch insert: {e}")
    finally:
        close_db_connection(conn, cursor)


if __name__ == '__main__':
    # Allow path override: python add_product_images_only.py app/tests/variants.json
    auto_batch_insert_from_file('app/tests/sherwani_wedding_top10_modified_json.json')
