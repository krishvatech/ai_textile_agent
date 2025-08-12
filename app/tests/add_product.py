from datetime import datetime, date
import os
import json
import enum

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.db.db_connection import get_db_connection, close_db_connection
from openai import OpenAI
from pinecone import Pinecone

# -------- ENV/CONFIG INIT --------
load_dotenv()
GPT_API_KEY = os.getenv("GPT_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "textile-products")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# -------- MODELS/CLIENTS INIT --------
_st_model = SentenceTransformer("clip-ViT-B-32")
openai_client = OpenAI(api_key=GPT_API_KEY)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pinecone_client.Index(PINECONE_INDEX)

# -------- ENUMS --------
class RentalStatus(str, enum.Enum):
    active = "active"
    returned = "returned"
    cancelled = "cancelled"

# -------- HELPERS --------

def parse_date_input(s: str) -> date:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Invalid date: {s}. Use YYYY-MM-DD or DD-MM-YYYY.")

def get_embedding(text):
    text = (text or "").strip()
    vec = _st_model.encode(text, normalize_embeddings=True)
    return vec.tolist()

def get_or_create_occasion_id(cursor, occasion_name):
    cursor.execute("SELECT id FROM occasions WHERE name = %s", (occasion_name,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        cursor.execute(
            "INSERT INTO occasions (name) VALUES (%s) RETURNING id",
            (occasion_name,)
        )
        occasion_id = cursor.fetchone()[0]
        print(f"Inserted new occasion '{occasion_name}' with ID {occasion_id}")
        return occasion_id

def upsert_variant_to_pinecone(variant_data):
    text_for_embedding = (
        f"{variant_data['name']} {variant_data['color']} {variant_data['fabric']} "
        f"{variant_data['category']} {variant_data.get('type','')} {variant_data.get('occasion','')} "
        f"{variant_data.get('description','')}"
    )
    embedding = get_embedding(text_for_embedding)

    def to_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    metadata = {
        "variant_id": variant_data["variant_id"],
        "product_id": variant_data["product_id"],
        "tenant_id": variant_data["tenant_id"],
        "name": variant_data["name"],
        "color": variant_data["color"],
        "size": variant_data["size"],
        "fabric": variant_data["fabric"],
        "category": variant_data["category"],
        "type": variant_data.get("type", ""),
        "occasion": variant_data.get("occasion", ""),
        "price": to_float(variant_data.get("price")),
        "rental_price": to_float(variant_data.get("rental_price")),
        "available_stock": int(variant_data.get("available_stock") or 0),
        "image_url": variant_data.get("image_url") or "",
        "is_rental": bool(variant_data.get("is_rental", False)),
        "is_active": bool(variant_data.get("is_active", True)),
        "description": variant_data.get("description", ""),
    }
    # Pinecone does not accept None values in metadata
    metadata = {k: v for k, v in metadata.items() if v is not None}

    pinecone_index.upsert(
        [(str(variant_data["variant_id"]), embedding, metadata)],
        namespace=PINECONE_NAMESPACE
    )
    print(f"Upserted variant {variant_data['variant_id']} to Pinecone.")

# -------- MAIN BATCH LOGIC --------

def auto_batch_insert_from_file(file_path):
    conn, cursor = get_db_connection()
    # NEW: load from a JSON array
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
    products = {}
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
        product_id_map = {}
        # Product insert
        for old_pid, prod in products.items():
            if not all([prod['tenant_id'], prod['name'], prod['category'], prod['type']]):
                print(f"Skipping product {old_pid} due to missing fields.")
                continue
            insert_product_query = '''
            INSERT INTO products (tenant_id, name, category, description, type, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
            '''
            cursor.execute(insert_product_query, (
                prod['tenant_id'], prod['name'], prod['category'], prod['description'], prod['type'],
                timestamp, timestamp
            ))
            new_pid = cursor.fetchone()[0]
            product_id_map[old_pid] = new_pid
            print(f"Inserted product '{prod['name']}' with new ID: {new_pid}")

        # Variant Insert
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
            is_rental = variant.get('is_rental', False)
            image_url = variant.get('image_url')
            occasions = variant.get('occasion', [])  # Array or str
            if isinstance(occasions, list):
                occasion_str = ' '.join(occasions)
            else:
                occasion_str = str(occasions)

            try:
                price_float = float(price)
                rental_price_float = float(rental_price) if rental_price not in (None, '', 'null') else None
                stock_int = int(available_stock) if available_stock is not None else 0
            except (ValueError, TypeError) as conv_err:
                print(f"Conversion error for variant {variant.get('product_name')}: {conv_err}")
                continue

            insert_variant_query = '''
            INSERT INTO product_variants (
                product_id, color, size, fabric, price, available_stock, 
                is_rental, rental_price, image_url, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
            '''
            cursor.execute(insert_variant_query, (
                new_pid, variant.get('color'), variant.get('size'), variant.get('fabric'),
                price_float, stock_int,
                is_rental, rental_price_float, image_url,
                timestamp, timestamp
            ))
            variant_id = cursor.fetchone()[0]
            print(f"Inserted variant '{variant.get('product_name')}' ({variant.get('color')}, {variant.get('size')}) with ID: {variant_id}")

            # Link occasions
            if isinstance(occasions, list):
                for occ in occasions:
                    occasion_id = get_or_create_occasion_id(cursor, occ)
                    cursor.execute(
                        '''INSERT INTO product_variant_occasions (variant_id, occasion_id)
                        VALUES (%s, %s)''',
                        (variant_id, occasion_id)
                    )
                    print(f"Linked variant {variant_id} to occasion '{occ}' (ID {occasion_id})")
            elif occasions:
                # handle single string occasion
                occasion_id = get_or_create_occasion_id(cursor, occasions)
                cursor.execute(
                    '''INSERT INTO product_variant_occasions (variant_id, occasion_id)
                    VALUES (%s, %s)''',
                    (variant_id, occasion_id)
                )
                print(f"Linked variant {variant_id} to occasion '{occasions}' (ID {occasion_id})")

            # Upsert to Pinecone
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
                "is_rental": is_rental,
                "is_active": is_active,
                "description": variant.get('description', '')
            }
            upsert_variant_to_pinecone(variant_data)

        conn.commit()
        print("All valid records automatically inserted into database and Pinecone.")
    except Exception as e:
        conn.rollback()
        print(f"Error during auto batch insert: {e}")
    finally:
        close_db_connection(conn, cursor)

if __name__ == '__main__':
    # Update file path as needed
    auto_batch_insert_from_file('app/tests/csvjson.json')
