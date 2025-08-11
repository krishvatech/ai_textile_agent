from datetime import datetime
from app.db.db_connection import get_db_connection, close_db_connection
import psycopg2  # if your get_db_connection uses psycopg2, else import your DB lib
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import enum

class RentalStatus(str, enum.Enum):
    active = "active"
    returned = "returned"
    cancelled = "cancelled"
load_dotenv()

api_key = os.getenv("GPT_API_KEY")
openai_client = OpenAI(api_key=api_key)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "textile-products")

client = Pinecone(api_key=PINECONE_API_KEY)
index = client.Index(INDEX_NAME)
print("index=",index)
def get_embedding(text):
    response = openai_client.embeddings.create(input=[text], model="text-embedding-3-small")
    return response.data[0].embedding
def get_or_create_occasion_id(cursor, occasion_name):
    # Check if occasion exists
    cursor.execute("SELECT id FROM occasions WHERE name = %s", (occasion_name,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        # Insert new occasion if not exists
        cursor.execute(
            "INSERT INTO occasions (name) VALUES (%s) RETURNING id",
            (occasion_name,)
        )
        occasion_id = cursor.fetchone()[0]
        print(f"Inserted new occasion '{occasion_name}' with ID {occasion_id}")
        return occasion_id

def upsert_variant_to_pinecone(variant_data):
    """
    variant_data: dict with keys
      variant_id, product_id, tenant_id, name, color, size, fabric, category,
      price, rental_price, available_stock, image_url, is_rental, is_active
    """
    text_for_embedding = f"{variant_data['name']} {variant_data['color']} {variant_data['fabric']} {variant_data['category']} {variant_data.get('type', '')} {variant_data.get('occasion', '')}"
    embedding = get_embedding(text_for_embedding)

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
        "price": float(variant_data["price"]) if variant_data["price"] else None,
        "rental_price": float(variant_data["rental_price"]) if variant_data["rental_price"] else None,
        "available_stock": variant_data["available_stock"],
        "image_url": variant_data["image_url"] if variant_data["image_url"] is not None else "",
        "is_rental": variant_data["is_rental"],
        "is_active": variant_data.get("is_active", True)
    }

    index.upsert([(str(variant_data["variant_id"]), embedding, metadata)])

def insert_product_and_variants():
    conn, cursor = get_db_connection()

    print("Enter Product details:")
    tenant_id = input("Tenant ID: ").strip()
    name = input("Product name: ").strip()
    category = input("Category: ").strip()
    description = input("Description: ").strip()
    type_ = input("Product type(Male/Female/Child): ").strip()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Insert Product first
    insert_product_query = '''
    INSERT INTO products (tenant_id, name, category, description,type, created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s,%s, %s) RETURNING id;
    '''
    cursor.execute(insert_product_query, (tenant_id, name, category, description, type_,timestamp, timestamp))
    product_id = cursor.fetchone()[0]
    conn.commit()

    print(f"Product inserted with ID: {product_id}")

    # Now ask for ProductVariant details
    add_more_variants = 'y'
    while add_more_variants.lower() == 'y':
        print("\nEnter Product Variant details:")
        color = input("Color: ").strip()
        size = input("Size: ").strip()
        occasion = input("Occasion (e.g. Wedding, Casual, Party): ").strip()
        fabric = input("Fabric: ").strip()
        price = input("Price: ").strip()
        available_stock = input("Available stock (integer): ").strip()
        is_rental_input = input("Is rental? (yes/no): ").strip().lower()
        is_rental = True if is_rental_input in ['yes', 'y'] else False
        rental_price = input("Rental price (optional): ").strip()
        image_url = input("Image URL (optional): ").strip()
        variant_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        insert_variant_query = '''
        INSERT INTO product_variants (
            product_id, color, size, fabric, price, available_stock, 
            is_rental, rental_price, image_url, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s , %s, %s, %s,%s, %s) RETURNING id;
        '''
        cursor.execute(insert_variant_query, (
            product_id, color, size, fabric,
            float(price) if price else None, 
            int(available_stock) if available_stock else 0,
            is_rental,
            float(rental_price) if rental_price else None,
            image_url if image_url else None,
            variant_timestamp,
            variant_timestamp
        ))
        variant_id = cursor.fetchone()[0]
        conn.commit()
        print(f"Product variant inserted with ID: {variant_id}")
        occasion_id = get_or_create_occasion_id(cursor, occasion)

        cursor.execute('''
            INSERT INTO product_variant_occasions (variant_id, occasion_id)
            VALUES (%s, %s)
        ''', (variant_id, occasion_id))
        conn.commit()
        print(f"Linked variant {variant_id} to occasion ID {occasion_id}")
        # Now immediately upsert this variant to Pinecone with embedding
        variant_data = {
            "variant_id": variant_id,
            "product_id": product_id,
            "tenant_id": tenant_id,
            "name": name,
            "color": color,
            "size": size,
            "fabric": fabric,
            "category": category,
            "type": type_,
            "occasion": occasion,
            "price": float(price) if price else None,
            "rental_price": float(rental_price) if rental_price else None,
            "available_stock": int(available_stock) if available_stock else 0,
            "image_url": image_url if image_url else None,
            "is_rental": is_rental,
            "is_active": True  # you can adjust if you want
        }
        upsert_variant_to_pinecone(variant_data)
        print("Product variant indexed in Pinecone.")
        if is_rental:
            print("This variant is rentable. Please enter rental availability periods.")
            add_rentals = 'y'
            while add_rentals.lower() == 'y':
                rental_start_date_str = input("Rental start date (YYYY-MM-DD): ").strip()
                rental_end_date_str = input("Rental end date (YYYY-MM-DD): ").strip()
                rental_price_str = input("Rental price (optional, press enter to use variant rental price): ").strip()
                rental_price = float(rental_price_str) if rental_price_str else variant_data["rental_price"]

                status = RentalStatus.active.value  # or 'booked' or your logic

                now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                insert_rental_query = '''
                INSERT INTO rentals (
                    product_variant_id, rental_start_date, rental_end_date, rental_price, status, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                '''

                cursor.execute(insert_rental_query, (
                    variant_id,
                    rental_start_date_str,
                    rental_end_date_str,
                    rental_price,
                    status,
                    now_str,
                    now_str
                ))
                rental_id = cursor.fetchone()[0]
                conn.commit()
                print(f"Inserted rental availability record with ID: {rental_id}")

                add_rentals = input("Add another rental availability period for this variant? (y/n): ").strip()
                add_more_variants = input("Add another variant? (y/n): ").strip()

    close_db_connection(conn, cursor)
    print("All data inserted and indexed. Connection closed.")

if __name__ == '__main__':
    insert_product_and_variants()
