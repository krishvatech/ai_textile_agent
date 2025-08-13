# ---------- IMPORTS, CONFIG, ETC. ----------

from app.db.db_connection import get_db_connection, close_db_connection
from pinecone import Pinecone
import os
from dotenv import load_dotenv

# --- ENVIRONMENT VARIABLES ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "textile-products")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pinecone_client.Index(PINECONE_INDEX)

# --- Occasion Helper ---
def get_or_create_occasion_id(cursor, occasion_name):
    if not occasion_name or not occasion_name.strip():
        return None
    cursor.execute("SELECT id FROM occasions WHERE name = %s", (occasion_name.strip(),))
    result = cursor.fetchone()
    if result:
        return result[0]
    cursor.execute("INSERT INTO occasions (name) VALUES (%s) RETURNING id", (occasion_name.strip(),))
    return cursor.fetchone()[0]

def update_all_tables(
    variant_id,
    product_updates={},
    variant_updates={},
    occasion_name=None
):
    conn, cursor = get_db_connection()
    try:
        # --- PRODUCT ID fetch ---
        cursor.execute("SELECT product_id FROM product_variants WHERE id = %s", (variant_id,))
        result = cursor.fetchone()
        if not result:
            print(f"❌ Variant {variant_id} not found.")
            return
        product_id = result[0]

        # --- PRODUCTS table update ---
        if product_updates:
            set_clause = ", ".join([f"{col} = %s" for col in product_updates])
            query = f"UPDATE products SET {set_clause} WHERE id = %s"
            cursor.execute(query, list(product_updates.values()) + [product_id])

        # --- PRODUCT_VARIANTS table update ---
        if variant_updates:
            set_clause = ", ".join([f"{col} = %s" for col in variant_updates])
            query = f"UPDATE product_variants SET {set_clause} WHERE id = %s"
            cursor.execute(query, list(variant_updates.values()) + [variant_id])

        # --- OCCASION MAPPING ---
        if occasion_name and occasion_name.strip():
            occasion_id = get_or_create_occasion_id(cursor, occasion_name)
            cursor.execute("DELETE FROM product_variant_occasions WHERE variant_id = %s", (variant_id,))
            cursor.execute(
                "INSERT INTO product_variant_occasions (variant_id, occasion_id) VALUES (%s, %s)",
                (variant_id, occasion_id)
            )

        conn.commit()

        # --- Pinecone metadata update (only metadata) ---
        new_metadata = {}
        new_metadata.update(product_updates)
        new_metadata.update(variant_updates)
        if occasion_name and occasion_name.strip():
            new_metadata['occasion'] = occasion_name.strip()
        new_metadata = {k: v for k, v in new_metadata.items() if v is not None}
        try:
            pinecone_index.update(id=str(variant_id), set_metadata=new_metadata, namespace=PINECONE_NAMESPACE)
            print(f"\n✅ Variant {variant_id} — all tables and Pinecone metadata successfully updated!")
        except Exception as e:
            print(f"Error updating Pinecone metadata for Variant {variant_id}: {e}")

    except Exception as e:
        conn.rollback()
        print(f"Error for Variant {variant_id}: {e}")
    finally:
        close_db_connection(conn, cursor)

if __name__ == '__main__':
    # Multiple variant ID input
    raw_input = input("Multiple Variant IDs dijiye (comma or space separated, e.g. 14,15,21): ").strip()
    ids = [x.strip() for x in raw_input.replace(',', ' ').split() if x.strip().isdigit()]
    variant_ids = [int(x) for x in ids]
    if not variant_ids:
        print("❌ Koi valid variant ID nahi diya.")
        exit(1)

    # --- PRODUCT TABLE FIELDS ---
    product_fields = [
        "name",
        "description",
        "category",
        "type"
    ]
    # --- VARIANT TABLE FIELDS (NEWLY ADDED: product_url) ---
    variant_fields = [
        "color",
        "size",
        "fabric",
        "price",
        "rental_price",
        "available_stock",
        "image_url",
        "is_active",
        "is_rental",
        "product_url"     # ⭐️ <-- Add this field for update/support
    ]

    print("\n--- Fields update karne ke liye value daalein (ye values sab variants par lagu hongi). Blank to skip ---")
    product_updates = {}
    for field in product_fields:
        user_in = input(f"{field}: ").strip()
        if user_in != "":
            product_updates[field] = user_in

    variant_updates = {}
    for field in variant_fields:
        user_in = input(f"{field}: ").strip()
        if user_in == "":
            continue
        if field in ("price", "rental_price"):
            try: variant_updates[field] = float(user_in)
            except: continue
        elif field == "available_stock":
            try: variant_updates[field] = int(user_in)
            except: continue
        elif field in ("is_active", "is_rental"):
            val = user_in.lower()
            if val in ("true", "1", "yes", "haan"):
                variant_updates[field] = True
            elif val in ("false", "0", "no", "nahi"):
                variant_updates[field] = False
        else:
            variant_updates[field] = user_in

    occasion_name = input("\nOccasion (e.g. Party/Wedding, blank to skip): ").strip()

    for variant_id in variant_ids:
        print(f"\n--- Updating Variant ID: {variant_id} ---")
        update_all_tables(
            variant_id=variant_id,
            product_updates=product_updates,
            variant_updates=variant_updates,
            occasion_name=occasion_name
        )
