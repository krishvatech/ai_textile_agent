# variant_sync.py

from app.db.db_connection import get_db_connection, close_db_connection
from pinecone import Pinecone
import os
from dotenv import load_dotenv

# --- LOAD ENVIRONMENT VARIABLES ---
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
        # Get product_id from variant
        cursor.execute("SELECT product_id FROM product_variants WHERE id = %s", (variant_id,))
        result = cursor.fetchone()
        if not result:
            print("❌ Variant not found.")
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

        # --- OCCASION Mapping ---
        if occasion_name and occasion_name.strip():
            occasion_id = get_or_create_occasion_id(cursor, occasion_name)
            cursor.execute("DELETE FROM product_variant_occasions WHERE variant_id = %s", (variant_id,))
            cursor.execute(
                "INSERT INTO product_variant_occasions (variant_id, occasion_id) VALUES (%s, %s)",
                (variant_id, occasion_id)
            )

        conn.commit()

        # --- Pinecone metadata update (only metadata, not embedding!) ---
        new_metadata = {}
        new_metadata.update(product_updates)
        new_metadata.update(variant_updates)
        if occasion_name and occasion_name.strip():
            new_metadata['occasion'] = occasion_name.strip()
        # Remove None values
        new_metadata = {k: v for k, v in new_metadata.items() if v is not None}

        # Metadata-only update (embedding/vector nahi chahiye)
        try:
            pinecone_index.update(id=str(variant_id), set_metadata=new_metadata, namespace=PINECONE_NAMESPACE)
            print("\n✅ All tables and Pinecone metadata successfully updated!")
        except Exception as e:
            print(f"Error updating Pinecone metadata: {e}")

    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
    finally:
        close_db_connection(conn, cursor)

if __name__ == '__main__':
    try:
        variant_id = int(input("Variant ID dijiye (e.g. 14): ").strip())
    except ValueError:
        print("❌ Galat ID format! Integer dijiye.")
        exit(1)

    # PRODUCT TABLE FIELDS
    product_fields = [
        "name",
        "description",
        "category",
        "type"
    ]
    variant_fields = [
        "color",
        "size",
        "fabric",
        "price",
        "rental_price",
        "available_stock",
        "image_url",
        "is_active",
        "is_rental"
    ]

    print("\n--- Products table fields (blank to skip) ---")
    product_updates = {}
    for field in product_fields:
        user_in = input(f"{field}: ").strip()
        if user_in != "":
            product_updates[field] = user_in

    print("\n--- Product_variants table fields (blank to skip) ---")
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

    update_all_tables(
        variant_id=variant_id,
        product_updates=product_updates,
        variant_updates=variant_updates,
        occasion_name=occasion_name
    )
