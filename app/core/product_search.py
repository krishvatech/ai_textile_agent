# import asyncio
# from app.vector.pinecone_client import get_index
# import openai
# import os
# from app.db.db_connection import get_db_connection, close_db_connection
# from dotenv import load_dotenv
# from app.vector.pinecone_client import get_image_index
# from ultralytics import solutions
# from app.core.image_clip import get_image_clip_embedding, get_text_clip_embedding
# from app.vector.pinecone_client import get_image_index
import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
import open_clip
import torch
import pprint

# ---------- Setup ----------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TEXT_INDEX_NAME = os.getenv("PINECONE_INDEX", "textile-products")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
pinecone = Pinecone(api_key=PINECONE_API_KEY)
device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')


def flexible_compare(a, b):
    # Handle booleans, numbers, strings
    if isinstance(a, bool) or isinstance(b, bool):
        def to_bool(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in {"true", "yes", "1"}
            return bool(val)
        return to_bool(a) == to_bool(b)
    return str(a) == str(b)

async def pinecone_fetch_records(entities: dict, tenant_id: int):
    """
    Capitalize all string entity values for Pinecone semantic search AND filtering.
    Always includes product_name in the result.
    """
    # Normalize entity values (capitalize strings)
    entities_cap = {
        k: str(v).capitalize() if isinstance(v, str) else v
        for k, v in entities.items()
    }
    texts = [str(v) for v in entities_cap.values() if isinstance(v, str) and v.strip()]
    print("Semantic search embedding text:", texts)
    if not texts:
        raise ValueError("entities must contain at least one non-empty string value for text-based search.")
    query_text = " ".join(texts)

    text_input = tokenizer([query_text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    query_vector = text_features[0].cpu().numpy().tolist()

    index = pinecone.Index(TEXT_INDEX_NAME)
    response = index.query(
        vector=query_vector,
        top_k=10,
        namespace=NAMESPACE,
        filter={"tenant_id": {"$eq": tenant_id}}
    )

    matches = []
    ids = [m.id for m in response.matches if hasattr(m, "id")]
    if ids:
        details = await fetch_records_by_ids(ids)
        for prod in details:
            item = {"id": prod.get("id"), "score": None}
            for m in response.matches:
                if hasattr(m, "id") and m.id == prod["id"]:
                    item["score"] = getattr(m, "score", None)
                    break
            item["product_name"] = prod.get("product_name")
            for key in entities_cap:
                item[key] = prod.get(key)
            matches.append(item)
    # Flexible filter using capitalized entities
    filtered_matches = [
    item for item in matches
        if all(
            # Only filter if the key is present in the product (not missing)
            key in item and flexible_compare(item.get(key), value)
            for key, value in entities_cap.items()
            if value not in [None, '', [], {}]
        )
    ]


    filtered_matches.sort(key=lambda x: x["score"], reverse=True)
    print("="*10)
    print(filtered_matches)
    print("="*10)
    return filtered_matches

async def fetch_records_by_ids(ids, index_name=TEXT_INDEX_NAME, namespace=NAMESPACE):
    """Fetch by ID: return metadata for each ID."""
    index = pinecone.Index(index_name)
    response = index.fetch(ids=ids, namespace=namespace)
    return [
        info.metadata | {"id": id_}
        for id_, info in response.vectors.items()
        if hasattr(info, "metadata") and info.metadata
    ]

# async def search_products_by_image(image_url, top_k=5, namespace=NAMESPACE):
#     img_emb = get_image_clip_embedding(image_url)  # This will be 512-dim if using ViT-B-32
#     index = get_image_index()
#     print('index :', index.describe_index_stats())
#     res = index.query(
#         vector=img_emb,
#         top_k=top_k,
#         include_metadata=True,
#         namespace=namespace
#     )
#     print('res..................122522', res)
#     products = []
#     for match in res.get("matches", []):
#         meta = match.get("metadata", {})
#         products.append({
#             "id": meta.get("id", "MISSING"),
#             "product_name": meta.get("product_name", "MISSING"),
#             "image_url": meta.get("image_url", "MISSING"),
#             "score": match.get("score", 0)
#         })
#     return products