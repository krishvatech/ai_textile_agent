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
    # 5 Sarees
    {
        "id": "P101",
        "tenant_id": 4,
        "sku": "SKU-S001",
        "product_name": "Banarasi Silk Saree",
        "category": "Saree",
        "fabric": "Silk",
        "color": "Red",
        "size": "Free Size",
        "price": 2999,
        "stock": "Available",
        "description": "Traditional red Banarasi silk saree with golden zari border.",
        "rental_price": 600,
        "is_rental": True,
        "available_stock": 10,
        "image_url": "https://cdn.example.com/saree1.jpg",
        "is_active": True
    },
    {
        "id": "P102",
        "tenant_id": 4,
        "sku": "SKU-S002",
        "product_name": "Kanjivaram Wedding Saree",
        "category": "Saree",
        "fabric": "Silk Blend",
        "color": "Magenta",
        "size": "Free Size",
        "price": 4499,
        "stock": "Available",
        "description": "Elegant magenta Kanjivaram saree, perfect for weddings.",
        "rental_price": None,
        "is_rental": False,
        "available_stock": 5,
        "image_url": "https://cdn.example.com/saree2.jpg",
        "is_active": True
    },
    {
        "id": "P103",
        "tenant_id": 4,
        "sku": "SKU-S003",
        "product_name": "Cotton Printed Saree",
        "category": "Saree",
        "fabric": "Cotton",
        "color": "Blue",
        "size": "Free Size",
        "price": 1299,
        "stock": "Available",
        "description": "Breathable blue cotton saree with floral prints.",
        "rental_price": None,
        "is_rental": False,
        "available_stock": 20,
        "image_url": "https://cdn.example.com/saree3.jpg",
        "is_active": True
    },
    {
        "id": "P104",
        "tenant_id": 4,
        "sku": "SKU-S004",
        "product_name": "Georgette Party Saree",
        "category": "Saree",
        "fabric": "Georgette",
        "color": "Peach",
        "size": "Free Size",
        "price": 1599,
        "stock": "Available",
        "description": "Peach georgette saree ideal for special occasions.",
        "rental_price": 300,
        "is_rental": True,
        "available_stock": 12,
        "image_url": "https://cdn.example.com/saree4.jpg",
        "is_active": True
    },
    {
        "id": "P105",
        "tenant_id": 4,
        "sku": "SKU-S005",
        "product_name": "Embroidered Net Saree",
        "category": "Saree",
        "fabric": "Net",
        "color": "Off White",
        "size": "Free Size",
        "price": 1899,
        "stock": "Not Available",
        "description": "Off white saree with sequin embroidered work.",
        "rental_price": None,
        "is_rental": False,
        "available_stock": 0,
        "image_url": "https://cdn.example.com/saree5.jpg",
        "is_active": False
    },
    # 5 Dresses
    {
        "id": "P106",
        "tenant_id": 4,
        "sku": "SKU-D001",
        "product_name": "Floral Maxi Dress",
        "category": "Dress",
        "fabric": "Rayon",
        "color": "Navy Blue",
        "size": "S, M, L, XL",
        "price": 1099,
        "stock": "Available",
        "description": "Navy blue floral maxi dress for summer.",
        "rental_price": 200,
        "is_rental": True,
        "available_stock": 8,
        "image_url": "https://cdn.example.com/dress1.jpg",
        "is_active": True
    },
    {
        "id": "P107",
        "tenant_id": 4,
        "sku": "SKU-D002",
        "product_name": "Embroidered Anarkali Dress",
        "category": "Dress",
        "fabric": "Georgette",
        "color": "Pink",
        "size": "M, L, XL",
        "price": 1799,
        "stock": "Available",
        "description": "Stunning pink embroidered anarkali dress.",
        "rental_price": 350,
        "is_rental": True,
        "available_stock": 4,
        "image_url": "https://cdn.example.com/dress2.jpg",
        "is_active": True
    },
    {
        "id": "P108",
        "tenant_id": 4,
        "sku": "SKU-D003",
        "product_name": "Casual Cotton Dress",
        "category": "Dress",
        "fabric": "Cotton",
        "color": "Yellow",
        "size": "S, M, L",
        "price": 799,
        "stock": "Available",
        "description": "Comfortable yellow knee-length cotton dress.",
        "rental_price": None,
        "is_rental": False,
        "available_stock": 10,
        "image_url": "https://cdn.example.com/dress3.jpg",
        "is_active": True
    },
    {
        "id": "P109",
        "tenant_id": 4,
        "sku": "SKU-D004",
        "product_name": "Party Wear Gown",
        "category": "Dress",
        "fabric": "Silk Blend",
        "color": "Green",
        "size": "L, XL",
        "price": 2299,
        "stock": "Available",
        "description": "Stylish green long gown for parties.",
        "rental_price": 400,
        "is_rental": True,
        "available_stock": 7,
        "image_url": "https://cdn.example.com/dress4.jpg",
        "is_active": True
    },
    {
        "id": "P110",
        "tenant_id": 4,
        "sku": "SKU-D005",
        "product_name": "Baby Doll Dress",
        "category": "Dress",
        "fabric": "Polyester",
        "color": "Purple",
        "size": "S, M",
        "price": 1299,
        "stock": "Not Available",
        "description": "Trendy purple baby doll style dress.",
        "rental_price": None,
        "is_rental": False,
        "available_stock": 0,
        "image_url": "https://cdn.example.com/dress5.jpg",
        "is_active": False
    },
    # 5 Cholis
    {
        "id": "P111",
        "tenant_id": 4,
        "sku": "SKU-C001",
        "product_name": "Mirror Work Choli",
        "category": "Choli",
        "fabric": "Cotton Blend",
        "color": "Pink",
        "size": "S, M, L",
        "price": 699,
        "stock": "Available",
        "description": "Trendy pink choli with mirror work.",
        "rental_price": 140,
        "is_rental": True,
        "available_stock": 9,
        "image_url": "https://cdn.example.com/choli1.jpg",
        "is_active": True
    },
    {
        "id": "P112",
        "tenant_id": 4,
        "sku": "SKU-C002",
        "product_name": "Velvet Embroidered Choli",
        "category": "Choli",
        "fabric": "Velvet",
        "color": "Bottle Green",
        "size": "M, L",
        "price": 999,
        "stock": "Available",
        "description": "Bottle green velvet choli with embroidery.",
        "rental_price": None,
        "is_rental": False,
        "available_stock": 6,
        "image_url": "https://cdn.example.com/choli2.jpg",
        "is_active": True
    },
    {
        "id": "P113",
        "tenant_id": 4,
        "sku": "SKU-C003",
        "product_name": "Silk Sleeveless Choli",
        "category": "Choli",
        "fabric": "Silk",
        "color": "Teal Blue",
        "size": "S, M",
        "price": 799,
        "stock": "Available",
        "description": "Teal blue silk sleeveless choli for navratri.",
        "rental_price": 120,
        "is_rental": True,
        "available_stock": 14,
        "image_url": "https://cdn.example.com/choli3.jpg",
        "is_active": True
    },
    {
        "id": "P114",
        "tenant_id": 4,
        "sku": "SKU-C004",
        "product_name": "Cotton Printed Choli",
        "category": "Choli",
        "fabric": "Cotton",
        "color": "Multicolor",
        "size": "M, L, XL",
        "price": 599,
        "stock": "Available",
        "description": "Colorful cotton choli with traditional Rajasthan print.",
        "rental_price": None,
        "is_rental": False,
        "available_stock": 17,
        "image_url": "https://cdn.example.com/choli4.jpg",
        "is_active": True
    },
    {
        "id": "P115",
        "tenant_id": 4,
        "sku": "SKU-C005",
        "product_name": "Designer Net Choli",
        "category": "Choli",
        "fabric": "Net",
        "color": "Black",
        "size": "S, M, L",
        "price": 849,
        "stock": "Not Available",
        "description": "Black designer choli with sequin work.",
        "rental_price": 180,
        "is_rental": True,
        "available_stock": 0,
        "image_url": "https://cdn.example.com/choli5.jpg",
        "is_active": False
    },
    # 5 Lehngas
    {
        "id": "P116",
        "tenant_id": 4,
        "sku": "SKU-L001",
        "product_name": "Bridal Silk Lehnga",
        "category": "Lehnga",
        "fabric": "Silk",
        "color": "Maroon",
        "size": "Free Size",
        "price": 10999,
        "stock": "Available",
        "description": "Heavy maroon bridal silk lehnga with intricate embroidery.",
        "rental_price": 2500,
        "is_rental": True,
        "available_stock": 2,
        "image_url": "https://cdn.example.com/lehnga1.jpg",
        "is_active": True
    },
    {
        "id": "P117",
        "tenant_id": 4,
        "sku": "SKU-L002",
        "product_name": "Designer Net Lehnga",
        "category": "Lehnga",
        "fabric": "Net",
        "color": "Beige",
        "size": "Free Size",
        "price": 4999,
        "stock": "Available",
        "description": "Beige net lehnga with sequins and beadwork.",
        "rental_price": 1200,
        "is_rental": True,
        "available_stock": 4,
        "image_url": "https://cdn.example.com/lehnga2.jpg",
        "is_active": True
    },
    {
        "id": "P118",
        "tenant_id": 4,
        "sku": "SKU-L003",
        "product_name": "Cotton Festive Lehnga",
        "category": "Lehnga",
        "fabric": "Cotton",
        "color": "Purple",
        "size": "S, M, L",
        "price": 2999,
        "stock": "Available",
        "description": "Purple cotton lehnga ideal for festive occasions.",
        "rental_price": None,
        "is_rental": False,
        "available_stock": 7,
        "image_url": "https://cdn.example.com/lehnga3.jpg",
        "is_active": True
    },
    {
        "id": "P119",
        "tenant_id": 4,
        "sku": "SKU-L004",
        "product_name": "Georgette Panelled Lehnga",
        "category": "Lehnga",
        "fabric": "Georgette",
        "color": "Yellow",
        "size": "Free Size",
        "price": 3999,
        "stock": "Available",
        "description": "Yellow georgette lehnga with mirror panel work.",
        "rental_price": 1000,
        "is_rental": True,
        "available_stock": 3,
        "image_url": "https://cdn.example.com/lehnga4.jpg",
        "is_active": True
    },
    {
        "id": "P120",
        "tenant_id": 4,
        "sku": "SKU-L005",
        "product_name": "Chiffon Embroidered Lehnga",
        "category": "Lehnga",
        "fabric": "Chiffon",
        "color": "Blue",
        "size": "M, L",
        "price": 3499,
        "stock": "Available",
        "description": "Blue chiffon lehnga with thread embroidery.",
        "rental_price": None,
        "is_rental": False,
        "available_stock": 5,
        "image_url": "https://cdn.example.com/lehnga5.jpg",
        "is_active": True
    },
    # 5 Shirts
    {
        "id": "P121",
        "tenant_id": 4,
        "sku": "SKU-T001",
        "product_name": "Men's Striped Shirt",
        "category": "Shirt",
        "fabric": "Cotton",
        "color": "White",
        "size": "39, 40, 42",
        "price": 799,
        "stock": "Available",
        "description": "Classic white and blue striped men's shirt.",
        "rental_price": None,
        "is_rental": False,
        "available_stock": 12,
        "image_url": "https://cdn.example.com/shirt1.jpg",
        "is_active": True
    },
    {
        "id": "P122",
        "tenant_id": 4,
        "sku": "SKU-T002",
        "product_name": "Slim Fit Casual Shirt",
        "category": "Shirt",
        "fabric": "Linen",
        "color": "Light Green",
        "size": "40, 42, 44",
        "price": 1099,
        "stock": "Available",
        "description": "Linen slim-fit light green casual shirt.",
        "rental_price": 180,
        "is_rental": True,
        "available_stock": 8,
        "image_url": "https://cdn.example.com/shirt2.jpg",
        "is_active": True
    },
    {
        "id": "P123",
        "tenant_id": 4,
        "sku": "SKU-T003",
        "product_name": "Formal Checked Shirt",
        "category": "Shirt",
        "fabric": "Polyester",
        "color": "Grey Checks",
        "size": "41, 42",
        "price": 899,
        "stock": "Not Available",
        "description": "Grey checked polyester formal shirt for office.",
        "rental_price": None,
        "is_rental": False,
        "available_stock": 0,
        "image_url": "https://cdn.example.com/shirt3.jpg",
        "is_active": False
    },
    {
        "id": "P124",
        "tenant_id": 4,
        "sku": "SKU-T004",
        "product_name": "Full Sleeve Denim Shirt",
        "category": "Shirt",
        "fabric": "Denim",
        "color": "Dark Blue",
        "size": "M, L, XL",
        "price": 1349,
        "stock": "Available",
        "description": "Dark blue denim full sleeve casual shirt.",
        "rental_price": None,
        "is_rental": False,
        "available_stock": 19,
        "image_url": "https://cdn.example.com/shirt4.jpg",
        "is_active": True
    },
    {
        "id": "P125",
        "tenant_id": 4,
        "sku": "SKU-T005",
        "product_name": "Designer Floral Shirt",
        "category": "Shirt",
        "fabric": "Cotton",
        "color": "Maroon",
        "size": "S, M, L, XL",
        "price": 999,
        "stock": "Available",
        "description": "Maroon and white floral printed designer shirt.",
        "rental_price": 160,
        "is_rental": True,
        "available_stock": 33,
        "image_url": "https://cdn.example.com/shirt5.jpg",
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
