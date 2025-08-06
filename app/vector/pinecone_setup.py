import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import requests
from PIL import Image
import torch
import open_clip

load_dotenv()

# --- Pinecone setup ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TEXT_INDEX_NAME = os.getenv("PINECONE_INDEX", "textile-products")
IMAGE_INDEX_NAME = os.getenv("PINECONE_IMAGE_INDEX", "textile-products-image")
NAMESPACE = os.getenv("PINECONE_NAMESPACE")

pinecone = Pinecone(api_key=PINECONE_API_KEY)

# --- open_clip setup ---
device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

def get_image_clip_embedding(image_url):
    img = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    img_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(img_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features[0].cpu().numpy().tolist()

def get_text_clip_embedding(text):
    text_input = tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features[0].cpu().numpy().tolist()

def clean_metadata(metadata):
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            if key == "rental_price":
                cleaned[key] = 0.0
            else:
                cleaned[key] = ""
        else:
            cleaned[key] = value
    return cleaned

product_data = [
    {
        "id": "P031",
        "tenant_id": 1,
        "sku": "SKU-P009",
        "product_name": "Formal Men's Trousers",
        "category": "Trousers",
        "fabric": "Polyester",
        "color": "Grey",
        "size": "32, 34, 36, 38",
        "price": 999,
        "stock": "Not Available",
        "description": "Grey formal trousers for office use.",
        "rental_price": None,
        "is_rental": False,
        "available_stock": 0,
        "image_url": "https://d2ki7eiqd260sq.cloudfront.net/Dark-Red-Pure-Organza-Banarasi-Saree3326c8f7-e149-4dc0-8d33-0bdb667ed43e.jpg",
        "is_active": True
    },
    {
        "id": "P032",
        "tenant_id": 3,
        "sku": "SKU-P010",
        "product_name": "Casual Men's Shirt",
        "category": "Shirts",
        "fabric": "Cotton",
        "color": "Blue",
        "size": "M, L, XL",
        "price": 799,
        "stock": "Not Available",
        "description": "Comfortable casual blue shirt.",
        "rental_price": None,
        "is_rental": False,
        "available_stock": 0,
        "image_url": "https://d2ki7eiqd260sq.cloudfront.net/Orange-Silk-Viscose-Embroidery-Saree23d3539f-7af8-4896-a47d-ef474d7ad90f.jpg",
        "is_active": True
    },
    {
        "id": "P033",
        "tenant_id": 2,
        "sku": "SKU-P011",
        "product_name": "Formal Women's Blazer",
        "category": "Blazers",
        "fabric": "Wool",
        "color": "Black",
        "size": "S, M, L",
        "price": 1999,
        "stock": "Not Available",
        "description": "Elegant black wool blazer.",
        "rental_price": None,
        "is_rental": False,
        "available_stock": 0,
        "image_url": "https://d2ki7eiqd260sq.cloudfront.net/Purple-Pochampalli-Pure-Cotton-Ikat-Saree44393cd5-a5cc-4e0f-b1ca-7952ca530715.jpg",
        "is_active": True
    }
]

# --- Delete and create indexes as 512-dim ---
for index_name in [TEXT_INDEX_NAME, IMAGE_INDEX_NAME]:
    idxs = [i["name"] for i in pinecone.list_indexes()]
    if index_name in idxs:
        print(f"Deleting old {index_name}")
        pinecone.delete_index(index_name)
    print(f"Creating Pinecone index: {index_name}")
    pinecone.create_index(
        name=index_name,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# --- 1. Index TEXT embeddings (512 dim, open_clip) ---
text_vectors = []
for item in product_data:
    txt_emb = get_text_clip_embedding(item["description"])
    print(f"Text Embedding for {item['id']}: {len(txt_emb)}")
    text_vectors.append({
        "id": item["id"],
        "values": txt_emb,
        "metadata": clean_metadata(item)
    })
pinecone.Index(TEXT_INDEX_NAME).upsert(vectors=text_vectors, namespace=NAMESPACE)
print(f"✅ Uploaded {len(text_vectors)} TEXT vectors to {TEXT_INDEX_NAME}")

# --- 2. Index IMAGE embeddings (512 dim, open_clip) ---
image_vectors = []
for item in product_data:
    image_url = item.get("image_url")
    if image_url:
        try:
            img_emb = get_image_clip_embedding(image_url)
            image_vectors.append({
                "id": item["id"],
                "values": img_emb,
                "metadata": clean_metadata(item)
            })
            print(f"Image Embedding for {item['id']}: {len(img_emb)}")
        except Exception as e:
            print(f"Image embedding failed for {item['id']} - {image_url}: {e}")

if image_vectors:
    pinecone.Index(IMAGE_INDEX_NAME).upsert(vectors=image_vectors, namespace=NAMESPACE)
    print(f"✅ Uploaded {len(image_vectors)} IMAGE vectors to {IMAGE_INDEX_NAME}")

print("🎉 All product data indexed for both TEXT and IMAGE search in Pinecone!")
