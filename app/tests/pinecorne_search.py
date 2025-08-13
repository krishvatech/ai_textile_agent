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
    if isinstance(a, bool) or isinstance(b, bool):
        def to_bool(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in {"true", "yes", "1"}
            return bool(val)
        return to_bool(a) == to_bool(b)
    return str(a) == str(b)

async def fetch_records_by_ids(ids, index_name=TEXT_INDEX_NAME, namespace=NAMESPACE):
    """Fetch by ID: return metadata for each ID."""
    index = pinecone.Index(index_name)
    response = index.fetch(ids=ids, namespace=namespace)
    return [
        info.metadata | {"id": id_}
        for id_, info in response.vectors.items()
        if hasattr(info, "metadata") and info.metadata
    ]

async def pinecone_fetch_records(entities: dict, tenant_id: int):
    """
    Multi-color search supported! If 'color' is given comma-separated, runs per-color, combines/dedupes results, sorts by score.
    """
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

    # ---------- MULTI-COLOR LOGIC ----------
    color_value = entities_cap.get("color", "")
    color_list = [c.strip().capitalize() for c in color_value.split(",") if c.strip()] if color_value else []
    all_results = []
    already_ids = set()
    index = pinecone.Index(TEXT_INDEX_NAME)

    if color_list:
        for color in color_list:
            filter_dict = {
                "tenant_id": {"$eq": tenant_id},
                "color": {"$eq": color}
            }
            print(f"Color sub-query filter: {filter_dict}")
            response = index.query(
                vector=query_vector,
                top_k=50,
                namespace=NAMESPACE,
                filter=filter_dict
            )
            ids = [m.id for m in response.matches if hasattr(m, "id")]
            if ids:
                details = await fetch_records_by_ids(ids)
                for prod in details:
                    pid = prod.get("id")
                    if pid and pid not in already_ids:
                        item = {"id": pid, "score": None}
                        for m in response.matches:
                            if hasattr(m, "id") and m.id == pid:
                                item["score"] = getattr(m, "score", None)
                                break
                        item["product_name"] = prod.get("product_name")
                        item["is_rental"] = prod.get("is_rental")
                        for key in entities_cap:
                            item[key] = prod.get(key)
                        all_results.append(item)
                        already_ids.add(pid)
    else:
        # No color: only normal filter
        filter_dict = {"tenant_id": {"$eq": tenant_id}}
        print(f"Normal query filter: {filter_dict}")
        response = index.query(
            vector=query_vector,
            top_k=50,
            namespace=NAMESPACE,
            filter=filter_dict
        )
        ids = [m.id for m in response.matches if hasattr(m, "id")]
        if ids:
            details = await fetch_records_by_ids(ids)
            for prod in details:
                pid = prod.get("id")
                if pid and pid not in already_ids:
                    item = {"id": pid, "score": None}
                    for m in response.matches:
                        if hasattr(m, "id") and m.id == pid:
                            item["score"] = getattr(m, "score", None)
                            break
                    item["product_name"] = prod.get("product_name")
                    item["is_rental"] = prod.get("is_rental")
                    for key in entities_cap:
                        item[key] = prod.get(key)
                    all_results.append(item)
                    already_ids.add(pid)

    # Sort by score descending
    all_results.sort(key=lambda x: x["score"] if x["score"] is not None else 0, reverse=True)
    print("="*10)
    # pprint.pprint(all_results)
    print("="*10)
    return all_results

def test_pinecone_fetch_records():
    test_cases = [
        {"entities": {'category': 'saree', 'color': 'red, Pink'}, "tenant_id": 4}
    ]
    for test_case in test_cases:
        print(f"\nTesting: {test_case}")
        try:
            results = asyncio.run(pinecone_fetch_records(test_case["entities"], test_case["tenant_id"]))
            if results:
                pprint.pprint(results)
            else:
                print("No matching products found.")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)

if __name__ == "__main__":
    test_pinecone_fetch_records()
