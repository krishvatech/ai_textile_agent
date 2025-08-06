import asyncio
from app.vector.pinecone_client import get_index
import openai
import os
from app.db.db_connection import get_db_connection, close_db_connection
from dotenv import load_dotenv
import psycopg2
from datetime import datetime
from app.vector.pinecone_client import get_image_index
from ultralytics import solutions
from app.core.image_clip import get_image_clip_embedding, get_text_clip_embedding
from app.vector.pinecone_client import get_image_index

load_dotenv()

IMAGE_INDEX_NAME = os.getenv("PINECONE_IMAGE_INDEX", "textile-products-image")
NAMESPACE = os.getenv("PINECONE_NAMESPACE")

OPENAI_API_KEY = os.getenv("GPT_API_KEY")
openai.api_key = OPENAI_API_KEY

IMAGE_INDEX_NAME = os.getenv("PINECONE_IMAGE_INDEX", "textile-products-image")

def get_embeddings(text):
    
    """
    Generate an embedding vector for the input text using OpenAI embeddings API.
    """
    # resp = openai.embeddings.create(input=[text], model="text-embedding-3-small")
    # return resp.data[0].embedding
    return get_text_clip_embedding(text)

def is_valid_record(meta):
    
    """
    Validate if a product record metadata is valid and should be included in results.
    Args:
        meta (dict): Metadata of a product from Pinecone.
    Returns:
        bool: True if valid, False if invalid (e.g., missing product name or inactive).
    """
    
    product_name = meta.get("product_name") or meta.get("name")
    is_active = meta.get("is_active", True)  # Default True if missing
    
    # Check for missing or invalid product name
    if not isinstance(product_name, str) or not product_name.strip() or product_name.strip().lower() == "missing":
        return False
    
    # Optionally exclude inactive records
    if not is_active:
        return False

    return True

def normalize_date(date_str):
    try:
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        return dt.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Date parse error: {e}")
        return None
    
def check_date_availability(conn, product_variant_id, requested_date):
    with conn.cursor() as cur:
        query = """
            SELECT COUNT(*)
            FROM public.rentals
            WHERE product_variant_id = %s
              AND status = 'active'
              AND %s BETWEEN rental_start_date AND rental_end_date
        """
        cur.execute(query, (product_variant_id, requested_date))
        count = cur.fetchone()[0]
        return count == 0  # True if available, False if booked

#for search product from pinecone
async def search_products(query=None,filters=None,top_k=10000,namespace=NAMESPACE):
    
    if query:
        embedding = get_embeddings(query)
    else:
        embedding = [0.0] * 1536  # zero vector for metadata-only search
    index = get_index()
    if not hasattr(index, 'query'):
        raise ValueError(f"Index is not queryable. Type: {type(index)}")

    pinecone_filter = filters or {}

    res = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        filter=pinecone_filter,
        namespace=namespace
    )
    products = []
    for match in res.get("matches", []):
        meta = match.get("metadata", {})
        if not is_valid_record(meta):
            print("Skipping invalid/incomplete record:", meta)
            continue
        products.append({
            "id": meta.get("id", "MISSING"),
            "tenant_id": meta.get("tenant_id", "MISSING"),
            "sku": meta.get("sku", "MISSING"),
            "product_name": meta.get("product_name", "MISSING"),
            "category": meta.get("category", "MISSING"),
            "fabric": meta.get("fabric", "MISSING"),
            "color": meta.get("color", "MISSING"),
            "size": meta.get("size", "MISSING"),
            "price": meta.get("price", "MISSING"),
            "stock": meta.get("stock", "MISSING"),
            "description": meta.get("description", "MISSING"),
            "rental_price": meta.get("rental_price", "MISSING"),
            "is_rental": meta.get("is_rental", "MISSING"),
            "available_stock": meta.get("available_stock", "MISSING"),
            "image_url": meta.get("image_url", "MISSING"),
            "is_active": meta.get("is_active", "MISSING")
        })
    return products

session_storage = {}

def get_session_state(session_id):
    """
    Retrieve or initialize session state (filters, tenant ID) for a given session ID.
    The session's stored filters and tenant ID.
    """
    if session_id not in session_storage:
        session_storage[session_id] = {
            "filters": {},
            "tenant_id": None
        }
    return session_storage[session_id]

async def get_unique_filters(namespace="__default__"):
    """
    Retrieve unique metadata values for filters (color, fabric, size, stock, rental status)
    from a sample of Pinecone index records.
    """
    
    index = get_index()
    dummy_vector = [0.0] * 1536

    res = index.query(
        vector=dummy_vector,
        top_k=1000,
        include_metadata=True,
        namespace=namespace
    )

    colors = set()
    fabrics = set()
    sizes = set()
    stocks = set()
    is_rentals = set()

    for match in res.get("matches", []):
        meta = match.get("metadata", {})
        if meta.get("color"):
            colors.add(meta["color"].capitalize())
        if meta.get("fabric"):
            fabrics.add(meta["fabric"].capitalize())
        if meta.get("size"):
            sizes.add(meta["size"].capitalize())
        if meta.get("stock"):
            stocks.add(meta["stock"].capitalize())
        if meta.get("is_rental") is not None:
            is_rentals.add(str(meta["is_rental"]).lower()) 

    return colors, fabrics, sizes, stocks, is_rentals

# Global cache variables 
ALL_COLORS = set()
ALL_FABRICS = set()
ALL_SIZES = set()
ALL_STOCKS = set()
ALL_IS_RENTALS = set()


async def load_metadata_cache():
    """
    Load and cache all unique filter metadata values globally for efficient access.
    Should be called once at startup or when refreshing metadata.
    """
    global ALL_COLORS, ALL_FABRICS, ALL_SIZES, ALL_STOCKS, ALL_IS_RENTALS
    ALL_COLORS, ALL_FABRICS, ALL_SIZES, ALL_STOCKS, ALL_IS_RENTALS = await get_unique_filters()


async def handle_user_input(session_id, entities=None, tenant_id=None, conn=None):
    """
    Handle user input by updating session filters and performing product search.
    Only the last key-value pair in 'entities' is applied as a filter update.
    """
    session = get_session_state(session_id)

    if tenant_id is not None:
        session["tenant_id"] = tenant_id
        session["filters"]["tenant_id"] = {"$eq": str(tenant_id)}

    # If entities is provided, directly set filters from it
    for key, value in entities.items():
        if value is None or value == "":
            continue
        if key == "product":
            session["filters"]["category"] = {"$eq": value.lower()}
        elif key == "color":
            session["filters"]["color"] = {"$eq": value.lower()}
        elif key == "fabric":
            session["filters"]["fabric"] = {"$eq": value.lower()}
        elif key == "size":
            session["filters"]["size"] = {"$eq": value.lower()}
        elif key == "stock":
            session["filters"]["stock"] = {"$eq": value.lower()}
        elif key == "occasion":
            session["filters"]["occasion"] = {"$eq": value.lower()}
        elif key == "is_rental":
            if value is True:
                session["filters"]["is_rental"] = {"$eq": True}
            elif value is False:
                session["filters"]["is_rental"] = {"$eq": False}
            else:
                session["filters"].pop("is_rental", None)

    results = await search_products(
        filters=session["filters"]
    )
    print("Before date filtering, product count:", len(results)) 
    rental_date_raw = entities.get("rental_date")
    rental_date = normalize_date(rental_date_raw) if rental_date_raw else None

    if rental_date and conn:
        available_results = []
        for product in results:
            variant_id = 14
            print("variant-id=",variant_id)
            if variant_id is None or variant_id == "MISSING":
                continue
            if check_date_availability(conn, variant_id, rental_date):
                available_results.append(product)
        results = available_results
        print("After date filtering, available count:", len(results))
    count = len(results)
    print("Count=", count)
    if count > 0:
        pass  
    else:
        print("No found")
    return results

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