import asyncio
from app.vector.pinecone_client import get_index
import openai
import os
import re  # For regex filter extraction
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("GPT_API_KEY")
openai.api_key = OPENAI_API_KEY


def get_embeddings(text):
    
    """
    Generate an embedding vector for the input text using OpenAI embeddings API.
    """
    resp = openai.embeddings.create(input=[text], model="text-embedding-3-small")
    return resp.data[0].embedding

def is_valid_record(meta):
    
    """
    Validate if a product record metadata is valid and should be included in results.
    Args:
        meta (dict): Metadata of a product from Pinecone.
    Returns:
        bool: True if valid, False if invalid (e.g., missing product name or inactive).
    """
    
    product_name = meta.get("product_name")
    is_active = meta.get("is_active", True)  # Default True if missing
    
    # Check for missing or invalid product name
    if not isinstance(product_name, str) or not product_name.strip() or product_name.strip().lower() == "missing":
        return False
    
    # Optionally exclude inactive records
    if not is_active:
        return False

    return True

#for search product from pinecone
async def search_products(query=None,filters=None,top_k=10000,namespace="ns1"):
    
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

async def get_unique_filters(namespace="ns1"):
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


async def handle_user_input(session_id,tenant_id=None, entities=None):
    """
    Handle user input by updating session filters and performing product search.
    Only the last key-value pair in 'entities' is applied as a filter update.
    """
    session = get_session_state(session_id)

    if tenant_id is not None:
        session["tenant_id"] = tenant_id
        session["filters"]["tenant_id"] = {"$eq": float(tenant_id)}

    # If entities is provided, directly set filters from it
    if entities:
        last_key = list(entities.keys())[-1]
        last_value = entities[last_key]

        if last_key == "product" and last_value:
            session["filters"]["category"] = {"$eq": last_value.capitalize()}
        elif last_key == "color" and last_value:
            session["filters"]["color"] = {"$eq": last_value.capitalize()}
        elif last_key == "fabric" and last_value:
            session["filters"]["fabric"] = {"$eq": last_value.capitalize()}
        elif last_key == "size" and last_value:
            session["filters"]["size"] = {"$eq": last_value.capitalize()}
        elif last_key == "stock" and last_value:
            session["filters"]["stock"] = {"$eq": last_value.capitalize()}
        elif last_key == "is_rental":
            if last_value is True:
                session["filters"]["is_rental"] = {"$eq": True}
            elif last_value is False:
                session["filters"]["is_rental"] = {"$eq": False}
            else:
                session["filters"].pop("is_rental", None)

    results = await search_products(
        filters=session["filters"]
    )
    count = len(results)
    print("Count=", count)
    if count > 0:
        pass  
    else:
        print("No found")
    return results



