# from datetime import datetime, date
# from variant_sync import update_variant_price, delete_variant
# import os
# import json
# import enum
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
# from app.db.db_connection import get_db_connection, close_db_connection
# from openai import OpenAI
# from pinecone import Pinecone

# # -------- ENV/CONFIG INIT --------
# load_dotenv()
# GPT_API_KEY = os.getenv("GPT_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX = os.getenv("PINECONE_INDEX", "textile-products")
# PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# # -------- MODELS/CLIENTS INIT --------
# _st_model = SentenceTransformer("clip-ViT-B-32")
# openai_client = OpenAI(api_key=GPT_API_KEY)
# pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
# pinecone_index = pinecone_client.Index(PINECONE_INDEX)

# # -------- ENUMS --------
# class RentalStatus(str, enum.Enum):
#     active = "active"
#     returned = "returned"
#     cancelled = "cancelled"

# # -------- HELPERS --------
# def get_embedding(text):
#     text = (text or "").strip()
#     vec = _st_model.encode(text, normalize_embeddings=True)
#     return vec.tolist()

# def get_or_create_occasion_id(cursor, occasion_name):
#     # Skip blank occasion string
#     if not occasion_name or not occasion_name.strip():
#         return None
#     cursor.execute("SELECT id FROM occasions WHERE name = %s", (occasion_name.strip(),))
#     result = cursor.fetchone()
#     if result:
#         return result[0]
#     else:
#         cursor.execute(
#             "INSERT INTO occasions (name) VALUES (%s) RETURNING id",
#             (occasion_name.strip(),)
#         )
#         occasion_id = cursor.fetchone()[0]
#         print(f"Inserted new occasion '{occasion_name}' with ID {occasion_id}")
#         return occasion_id

# def upsert_variant_to_pinecone(variant_data):
#     text_for_embedding = (
#         f"{variant_data['name']} {variant_data['color']} {variant_data['fabric']} "
#         f"{variant_data['category']} {variant_data.get('type','')} {variant_data.get('occasion','')} "
#         f"{variant_data.get('description','')}"
#     )
#     embedding = get_embedding(text_for_embedding)

#     def to_float(v):
#         try:
#             return float(v)
#         except (TypeError, ValueError):
#             return None

#     metadata = {
#         "variant_id": variant_data["variant_id"],
#         "product_id": variant_data["product_id"],
#         "tenant_id": variant_data["tenant_id"],
#         "name": variant_data["name"],
#         "color": variant_data["color"],
#         "size": variant_data["size"],
#         "fabric": variant_data["fabric"],
#         "category": variant_data["category"],
#         "type": variant_data.get("type", ""),
#         "occasion": variant_data.get("occasion", ""),
#         "price": to_float(variant_data.get("price")),
#         "rental_price": to_float(variant_data.get("rental_price")),
#         "available_stock": int(variant_data.get("available_stock") or 0),
#         "image_url": variant_data.get("image_url") or "",
#         "product_url": variant_data.get("product_url") or "",
#         "is_rental": bool(variant_data.get("is_rental", False)),
#         "is_active": bool(variant_data.get("is_active", True)),
#         "description": variant_data.get("description", ""),
#     }
#     metadata = {k: v for k, v in metadata.items() if v is not None}

#     pinecone_index.upsert(
#         [(str(variant_data["variant_id"]), embedding, metadata)],
#         namespace=PINECONE_NAMESPACE
#     )
#     print(f"Upserted variant {variant_data['variant_id']} to Pinecone.")

# # -------- MAIN BATCH LOGIC --------
# def auto_batch_insert_from_file(file_path):
#     conn, cursor = get_db_connection()
#     with open(file_path, 'r', encoding='utf-8') as f:
#         try:
#             data = json.load(f)
#         except Exception as e:
#             print(f"Could not parse JSON: {e}")
#             close_db_connection(conn, cursor)
#             return

#     if not data or not isinstance(data, list):
#         print("No valid records found in file (expecting JSON array). Exiting.")
#         close_db_connection(conn, cursor)
#         return

#     # Group unique products
#     products = {}
#     for variant in data:
#         pid = variant.get('product_id')
#         if pid and pid not in products:
#             products[pid] = {
#                 'tenant_id': variant.get('tenant_id'),
#                 'name': variant.get('product_name', ''),
#                 'category': variant.get('category', ''),
#                 'description': variant.get('description', ''),
#                 'type': variant.get('type', ''),
#             }

#     try:
#         timestamp = datetime.now()
#         product_id_map = {}
#         # Product insert
#         for old_pid, prod in products.items():
#             if not all([prod['tenant_id'], prod['name'], prod['category'], prod['type']]):
#                 print(f"Skipping product {old_pid} due to missing fields.")
#                 continue
#             insert_product_query = '''
#             INSERT INTO products (tenant_id, name, category, description, type, created_at, updated_at)
#             VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
#             '''
#             cursor.execute(insert_product_query, (
#                 prod['tenant_id'], prod['name'], prod['category'], prod['description'], prod['type'],
#                 timestamp, timestamp
#             ))
#             new_pid = cursor.fetchone()[0]
#             product_id_map[old_pid] = new_pid
#             print(f"Inserted product '{prod['name']}' with new ID: {new_pid}")

#         # Variant Insert
#         for variant in data:
#             old_pid = variant.get('product_id')
#             if old_pid not in product_id_map:
#                 print(f"Skipping variant due to missing product ID: {old_pid}")
#                 continue
#             new_pid = product_id_map[old_pid]
#             price = variant.get('price')
#             rental_price = variant.get('rental_price')
#             available_stock = variant.get('available_stock', 0)
#             if price is None:
#                 print(f"Skipping variant due to missing price: {variant.get('product_name')}")
#                 continue

#             is_active = not variant.get('extra', {}).get('is_demo', False)
#             is_rental_raw = variant.get('is_rental', False)
#             if isinstance(is_rental_raw, str):
#                 is_rental = is_rental_raw.lower() in ('true', '1', 'yes')
#             else:
#                 is_rental = bool(is_rental_raw)

#             image_url = variant.get('image_url')
#             product_url = variant.get('product_url')

#             # Occasion field fix: always treat as list
#             occasions_raw = variant.get('occasion', [])
#             if isinstance(occasions_raw, list):
#                 occasions = [o for o in occasions_raw if o]    # remove blanks/nulls
#             elif isinstance(occasions_raw, str) and occasions_raw.strip():
#                 occasions = [occasions_raw.strip()]
#             else:
#                 occasions = []
#             print(f"Processing occasions for variant '{variant.get('product_name')}': {occasions}")

#             occasion_str = ' '.join(occasions) if occasions else ''

#             try:
#                 price_float = float(price)
#                 rental_price_float = float(rental_price) if rental_price not in (None, '', 'null') else None
#                 stock_int = int(available_stock) if available_stock is not None else 0
#             except (ValueError, TypeError) as conv_err:
#                 print(f"Conversion error for variant {variant.get('product_name')}: {conv_err}")
#                 continue

#             insert_variant_query = '''
#             INSERT INTO product_variants (
#                 product_id, color, size, fabric, price, available_stock, 
#                 is_rental, rental_price, image_url, created_at, updated_at, product_url
#             ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
#             '''
#             cursor.execute(insert_variant_query, (
#                 new_pid, variant.get('color'), variant.get('size'), variant.get('fabric'),
#                 price_float, stock_int,
#                 is_rental, rental_price_float, image_url,
#                 timestamp, timestamp, product_url
#             ))
#             variant_id = cursor.fetchone()[0]
#             print(f"Inserted variant '{variant.get('product_name')}' ({variant.get('color')}, {variant.get('size')}) with ID: {variant_id}")

#             # LINK OCCASIONS
#             for occ in occasions:
#                 if not occ: continue  # Skip blank values
#                 occasion_id = get_or_create_occasion_id(cursor, occ)
#                 if not occasion_id:
#                     print(f"Skipped blank/invalid occasion for: {occ}")
#                     continue
#                 cursor.execute(
#                     '''INSERT INTO product_variant_occasions (variant_id, occasion_id)
#                     VALUES (%s, %s)''',
#                     (variant_id, occasion_id)
#                 )
#                 print(f"Linked variant {variant_id} to occasion '{occ}' (ID {occasion_id})")

#             # Upsert to Pinecone
#             variant_data = {
#                 "variant_id": variant_id,
#                 "product_id": new_pid,
#                 "tenant_id": variant.get('tenant_id'),
#                 "name": variant.get('product_name', ''),
#                 "color": variant.get('color', ''),
#                 "size": variant.get('size', ''),
#                 "fabric": variant.get('fabric', ''),
#                 "category": variant.get('category', ''),
#                 "type": variant.get('type', ''),
#                 "occasion": occasion_str,
#                 "price": price_float,
#                 "rental_price": rental_price_float,
#                 "available_stock": stock_int,
#                 "image_url": image_url if image_url is not None else "",
#                 "product_url": product_url if product_url is not None else "",
#                 "is_rental": is_rental,
#                 "is_active": is_active,
#                 "description": variant.get('description', '')
#             }
#             upsert_variant_to_pinecone(variant_data)

#         conn.commit()
#         print("All valid records automatically inserted into database and Pinecone.")
#     except Exception as e:
#         conn.rollback()
#         print(f"Error during auto batch insert: {e}")
#     finally:
#         close_db_connection(conn, cursor)

# if __name__ == '__main__':
#     # Replace path as needed
#     auto_batch_insert_from_file('app/tests/blouses_rental_variants.json')


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
from urllib.parse import urlsplit, urlunsplit, quote

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
# ID scheme controls how vector IDs are formed:
#   composite (default): img:{tenant_id}:{product_id}:{variant_id}:{idx}:{hash}
#   variant:            img:{variant_id}  (use with USE_FIRST_IMAGE_ONLY=true)
#   variant_idx:        img:{variant_id}:{idx}
IMAGE_ID_SCHEME = os.getenv("IMAGE_ID_SCHEME", "composite").lower()
USE_FIRST_IMAGE_ONLY = os.getenv("USE_FIRST_IMAGE_ONLY", "false").strip().lower() in {"1","true","yes"}

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
    # Add browser-y headers; some CDNs block default Python UA
    parts = urlsplit(url)
    referer = f"{parts.scheme}://{parts.netloc}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": referer,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def _get(u: str):
        r = requests.get(u, headers=headers, timeout=20)
        if r.status_code == 403 and "+" in u:
            # Retry with '+' percent-encoded, which some servers require in PATH
            enc_path = quote(parts.path, safe="/:@")  # encodes '+' -> %2B
            u2 = urlunsplit((parts.scheme, parts.netloc, enc_path, parts.query, parts.fragment))
            r = requests.get(u2, headers=headers, timeout=20)
        r.raise_for_status()
        return r

    r = _get(url)
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
    urls = urls[:1]
    # Optionally limit to first image only (useful when IMAGE_ID_SCHEME='variant')
    if USE_FIRST_IMAGE_ONLY and urls:
        urls = [urls[0]]

    if not urls:
        print(f"[img-skip] variant {variant_id}: no image URL found")
        return

    for idx, url in enumerate(urls):
        try:
            img_vec = image_embed_from_url(url)
        except Exception as e:
            print(f"[warn] image embed failed for variant {variant_id} url={url}: {e}")
            continue

        # Build vector ID according to configured scheme
        img_vec_id = str(variant_id)
        img_meta = {
            **{k: v for k, v in _common_metadata(variant_data).items() if v is not None},
            # "modality": "image",  # removed per user request
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
    # auto_batch_insert_from_file('app/tests/sarees_partywear_top10_modified_json.json')
    # auto_batch_insert_from_file('app/tests/sarees_wedding_top10_modified_json.json')
    # auto_batch_insert_from_file('app/tests/salwar_kameez_wedding_top10_modified.json')
    # auto_batch_insert_from_file('app/tests/salwar_kameez_reception_top10_modified.json')
    # auto_batch_insert_from_file('app/tests/salwar_kameez_partywear_top10_modified.json')
    # auto_batch_insert_from_file('app/tests/salwar_kameez_diwali_all_products_modified.json')
    # auto_batch_insert_from_file('app/tests/lehenga_choli_partywear_all_products_modified.json')
    # auto_batch_insert_from_file('app/tests/lehenga_choli_navratri_all_products.json')
    # auto_batch_insert_from_file('app/tests/lehenga_choli_all_products_modified.json')
    # auto_batch_insert_from_file('app/tests/kurtis_festival_top10_modified_json.json')
    # auto_batch_insert_from_file('app/tests/kurtis_casual_top10_modified_json.json')
    # auto_batch_insert_from_file('app/tests/jodhpuri_wedding_top10.json')
    # auto_batch_insert_from_file('app/tests/gown_partywear_top10_modified_json.json')
    # auto_batch_insert_from_file('app/tests/products_category_pathani_set.json')
    # auto_batch_insert_from_file('app/tests/products_category_saree.json')
    # auto_batch_insert_from_file('app/tests/products_category_salwar_kameez.json')
    # auto_batch_insert_from_file('app/tests/products_category_kurta_set.json')
    # auto_batch_insert_from_file('app/tests/products_category_gown.json')
    # auto_batch_insert_from_file('app/tests/products_category_chaniya_choli.json')
    auto_batch_insert_from_file('app/tests/sherwani_groom_top10_modified_json.json')
    auto_batch_insert_from_file('app/tests/sherwani_wedding_top10_modified_json.json')