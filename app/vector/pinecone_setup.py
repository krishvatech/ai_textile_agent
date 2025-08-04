import os 
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "textile-products")
NAMESPACE = os.getenv("PINECONE_NAMESPACE")

pinecone = Pinecone(api_key=PINECONE_API_KEY)
client = OpenAI(api_key=os.getenv("GPT_API_KEY"))

# product_data = [
#     {"id": "P001", "tenant_id": 1, "sku": "SKU-P001", "product_name": "Banarasi Silk Saree", "category": "Saree", "fabric": "Silk", "color": "Red-Gold", "size": "Free Size", "price": 2999, "stock": "Available", "description": "Elegant Banarasi silk saree with zari work.", "rental_price": 1499.5, "is_rental": True, "available_stock": 10, "image_url": "https://example.com/images/p001.jpg", "is_active": True},
#     {"id": "P002", "tenant_id": 2, "sku": "SKU-P002", "product_name": "Chiffon Party Wear Saree", "category": "Saree", "fabric": "Chiffon", "color": "Blue", "size": "Free Size", "price": 1599, "stock": "Available", "description": "Lightweight blue chiffon saree ideal for parties.", "rental_price": None, "is_rental": False, "available_stock": 15, "image_url": "https://example.com/images/p002.jpg", "is_active": True},
#     {"id": "P003", "tenant_id": 3, "sku": "SKU-P003", "product_name": "Cotton Daily Wear Kurti", "category": "Kurti", "fabric": "Cotton", "color": "White", "size": "M, L, XL", "price": 499, "stock": "Available", "description": "Comfortable white kurti for everyday wear.", "rental_price": 249.0, "is_rental": True, "available_stock": 20, "image_url": "https://example.com/images/p003.jpg", "is_active": True},
#     {"id": "P004", "tenant_id": 4, "sku": "SKU-P004", "product_name": "Rayon Printed Kurti", "category": "Kurti", "fabric": "Rayon", "color": "Yellow", "size": "S, M, L", "price": 599, "stock": "Available", "description": "Yellow rayon kurti with floral prints.", "rental_price": None, "is_rental": False, "available_stock": 12, "image_url": "https://example.com/images/p004.jpg", "is_active": True},
#     {"id": "P005", "tenant_id": 5, "sku": "SKU-P005", "product_name": "Designer Lehenga Choli", "category": "Lehenga", "fabric": "Net", "color": "Pink", "size": "Free Size", "price": 4999, "stock": "Available", "description": "Pink designer lehenga with detailed embroidery.", "rental_price": 2499.0, "is_rental": True, "available_stock": 5, "image_url": "https://example.com/images/p005.jpg", "is_active": True},
#     {"id": "P006", "tenant_id": 1, "sku": "SKU-P006", "product_name": "Men's Cotton Shirt", "category": "Shirt", "fabric": "Cotton", "color": "Light Blue", "size": "M, L, XL, XXL", "price": 799, "stock": "Available", "description": "Light blue shirt perfect for summer.", "rental_price": None, "is_rental": False, "available_stock": 25, "image_url": "https://example.com/images/p006.jpg", "is_active": True},
#     {"id": "P007", "tenant_id": 2, "sku": "SKU-P007", "product_name": "Georgette Anarkali Suit", "category": "Suit", "fabric": "Georgette", "color": "Green", "size": "M, L, XL", "price": 1799, "stock": "Available", "description": "Flowy green georgette anarkali suit.", "rental_price": 899.5, "is_rental": True, "available_stock": 8, "image_url": "https://example.com/images/p007.jpg", "is_active": True},
#     {"id": "P008", "tenant_id": 3, "sku": "SKU-P008", "product_name": "Casual Palazzo Pants", "category": "Palazzo", "fabric": "Cotton", "color": "Black", "size": "S, M, L, XL", "price": 399, "stock": "Available", "description": "Stretchy cotton palazzo for daily wear.", "rental_price": None, "is_rental": False, "available_stock": 30, "image_url": "https://example.com/images/p008.jpg", "is_active": True},
#     {"id": "P009", "tenant_id": 4, "sku": "SKU-P009", "product_name": "Formal Men's Trousers", "category": "Trousers", "fabric": "Polyester", "color": "Grey", "size": "32, 34, 36, 38", "price": 999, "stock": "Available", "description": "Grey formal trousers for office use.", "rental_price": None, "is_rental": False, "available_stock": 18, "image_url": "https://example.com/images/p009.jpg", "is_active": True},
#     {"id": "P010", "tenant_id": 5, "sku": "SKU-P010", "product_name": "Girls Frock Dress", "category": "Dress", "fabric": "Net", "color": "Peach", "size": "2-6 Years", "price": 599, "stock": "Available", "description": "Peach net frock for girls, party wear.", "rental_price": 299.0, "is_rental": True, "available_stock": 14, "image_url": "https://example.com/images/p010.jpg", "is_active": True},
#     {"id": "P011", "tenant_id": 1, "sku": "SKU-P011", "product_name": "Men's Denim Jeans", "category": "Jeans", "fabric": "Denim", "color": "Blue", "size": "30, 32, 34, 36", "price": 1299, "stock": "Available", "description": "Classic blue denim jeans for men.", "rental_price": None, "is_rental": False, "available_stock": 22, "image_url": "https://example.com/images/p011.jpg", "is_active": True},
#     {"id": "P012", "tenant_id": 2, "sku": "SKU-P012", "product_name": "Women's Printed Skirt", "category": "Skirt", "fabric": "Cotton", "color": "Multi", "size": "S, M, L", "price": 749, "stock": "Available", "description": "Colorful printed cotton skirt.", "rental_price": None, "is_rental": False, "available_stock": 16, "image_url": "https://example.com/images/p012.jpg", "is_active": True},
#     {"id": "P013", "tenant_id": 3, "sku": "SKU-P013", "product_name": "Silk Dupatta", "category": "Dupatta", "fabric": "Silk", "color": "Maroon", "size": "2.5m", "price": 399, "stock": "Available", "description": "Soft silk maroon dupatta.", "rental_price": 199.5, "is_rental": True, "available_stock": 25, "image_url": "https://example.com/images/p013.jpg", "is_active": True},
#     {"id": "P014", "tenant_id": 4, "sku": "SKU-P014", "product_name": "Kids Cotton Shorts", "category": "Shorts", "fabric": "Cotton", "color": "Navy", "size": "2-10 Years", "price": 299, "stock": "Available", "description": "Comfortable navy shorts for kids.", "rental_price": None, "is_rental": False, "available_stock": 35, "image_url": "https://example.com/images/p014.jpg", "is_active": True},
#     {"id": "P015", "tenant_id": 5, "sku": "SKU-P015", "product_name": "Woolen Winter Shawl", "category": "Shawl", "fabric": "Wool", "color": "Brown", "size": "Free Size", "price": 899, "stock": "Available", "description": "Brown woolen shawl for winter.", "rental_price": None, "is_rental": False, "available_stock": 10, "image_url": "https://example.com/images/p015.jpg", "is_active": True},
#     {"id": "P016", "tenant_id": 1, "sku": "SKU-P016", "product_name": "Chanderi Silk Dupatta", "category": "Dupatta", "fabric": "Chanderi Silk", "color": "Off White", "size": "2.3m", "price": 549, "stock": "Available", "description": "Elegant Chanderi silk dupatta.", "rental_price": 274.5, "is_rental": True, "available_stock": 12, "image_url": "https://example.com/images/p016.jpg", "is_active": True},
#     {"id": "P017", "tenant_id": 2, "sku": "SKU-P017", "product_name": "Fancy Lycra Leggings", "category": "Leggings", "fabric": "Lycra", "color": "Pink", "size": "S, M, L, XL", "price": 349, "stock": "Available", "description": "Pink lycra leggings, stretchable.", "rental_price": None, "is_rental": False, "available_stock": 28, "image_url": "https://example.com/images/p017.jpg", "is_active": True},
#     {"id": "P018", "tenant_id": 3, "sku": "SKU-P018", "product_name": "Embroidered Kurta Set", "category": "Kurta Set", "fabric": "Cotton Blend", "color": "Grey", "size": "M, L, XL", "price": 999, "stock": "Available", "description": "Grey embroidered kurta set with pants.", "rental_price": 499.0, "is_rental": True, "available_stock": 9, "image_url": "https://example.com/images/p018.jpg", "is_active": True},
#     {"id": "P019", "tenant_id": 4, "sku": "SKU-P019", "product_name": "Printed Night Suit", "category": "Nightwear", "fabric": "Cotton", "color": "White-Blue", "size": "M, L, XL", "price": 649, "stock": "Available", "description": "Soft printed night suit set.", "rental_price": None, "is_rental": False, "available_stock": 20, "image_url": "https://example.com/images/p019.jpg", "is_active": True},
#     {"id": "P020", "tenant_id": 5, "sku": "SKU-P020", "product_name": "Boys Casual T-Shirt", "category": "T-Shirt", "fabric": "Cotton", "color": "Green", "size": "4-12 Years", "price": 349, "stock": "Available", "description": "Cool green t-shirt for boys.", "rental_price": None, "is_rental": False, "available_stock": 40, "image_url": "https://example.com/images/p020.jpg", "is_active": True},
#     {"id": "P021", "tenant_id": 1, "sku": "SKU-P021", "product_name": "Men's Kurta Pajama", "category": "Kurta Pajama", "fabric": "Cotton", "color": "Beige", "size": "M, L, XL, XXL", "price": 1099, "stock": "Available", "description": "Traditional beige kurta pajama.", "rental_price": 549.5, "is_rental": True, "available_stock": 7, "image_url": "https://example.com/images/p021.jpg", "is_active": True},
#     {"id": "P022", "tenant_id": 2, "sku": "SKU-P022", "product_name": "Women’s Long Shrug", "category": "Shrug", "fabric": "Rayon", "color": "Black", "size": "S, M, L", "price": 499, "stock": "Available", "description": "Stylish black rayon shrug.", "rental_price": None, "is_rental": False, "available_stock": 18, "image_url": "https://example.com/images/p022.jpg", "is_active": True},
#     {"id": "P023", "tenant_id": 3, "sku": "SKU-P023", "product_name": "Girls Embroidered Top", "category": "Top", "fabric": "Cotton", "color": "Yellow", "size": "8-14 Years", "price": 399, "stock": "Available", "description": "Bright yellow top with embroidery.", "rental_price": None, "is_rental": False, "available_stock": 22, "image_url": "https://example.com/images/p023.jpg", "is_active": True},
#     {"id": "P024", "tenant_id": 4, "sku": "SKU-P024", "product_name": "Cotton Saree with Blouse", "category": "Saree", "fabric": "Cotton", "color": "Sky Blue", "size": "Free Size", "price": 999, "stock": "Available", "description": "Sky blue cotton saree with matching blouse.", "rental_price": 499.0, "is_rental": True, "available_stock": 11, "image_url": "https://example.com/images/p024.jpg", "is_active": True},
#     {"id": "P025", "tenant_id": 5, "sku": "SKU-P025", "product_name": "Velvet Party Blazer", "category": "Blazer", "fabric": "Velvet", "color": "Dark Blue", "size": "M, L, XL", "price": 2299, "stock": "Available", "description": "Dark blue velvet blazer for parties.", "rental_price": None, "is_rental": False, "available_stock": 6, "image_url": "https://example.com/images/p025.jpg", "is_active": True},
#     {"id": "P026", "tenant_id": 1, "sku": "SKU-P026", "product_name": "Boys Denim Jacket", "category": "Jacket", "fabric": "Denim", "color": "Blue", "size": "6-16 Years", "price": 1199, "stock": "Available", "description": "Trendy blue denim jacket for boys.", "rental_price": None, "is_rental": False, "available_stock": 13, "image_url": "https://example.com/images/p026.jpg", "is_active": True},
#     {"id": "P027", "tenant_id": 2, "sku": "SKU-P027", "product_name": "Printed Rayon Gown", "category": "Gown", "fabric": "Rayon", "color": "Purple", "size": "M, L, XL", "price": 1499, "stock": "Available", "description": "Purple rayon gown with prints.", "rental_price": 749.5, "is_rental": True, "available_stock": 9, "image_url": "https://example.com/images/p027.jpg", "is_active": True},
#     {"id": "P028", "tenant_id": 3, "sku": "SKU-P028", "product_name": "Men's Woolen Sweater", "category": "Sweater", "fabric": "Wool", "color": "Grey", "size": "M, L, XL, XXL", "price": 899, "stock": "Available", "description": "Warm grey woolen sweater for men.", "rental_price": None, "is_rental": False, "available_stock": 17, "image_url": "https://example.com/images/p028.jpg", "is_active": True},
#     {"id": "P029", "tenant_id": 4, "sku": "SKU-P029", "product_name": "Girls Net Party Dress", "category": "Dress", "fabric": "Net", "color": "Pink", "size": "3-8 Years", "price": 799, "stock": "Available", "description": "Pink net party dress for girls.", "rental_price": 399.5, "is_rental": True, "available_stock": 14, "image_url": "https://example.com/images/p029.jpg", "is_active": True},
#     {"id": "P030", "tenant_id": 5, "sku": "SKU-P030", "product_name": "Men's Formal Suit", "category": "Suit", "fabric": "Polyester", "color": "Black", "size": "M, L, XL, XXL", "price": 3999, "stock": "Available", "description": "Classic black suit for formal occasions.", "rental_price": None, "is_rental": False, "available_stock": 4, "image_url": "https://example.com/images/p030.jpg", "is_active": True}
# ]
product_data=[
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
                    "image_url": "https://example.com/images/p009.jpg",
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
                "image_url": "https://example.com/images/p010.jpg",
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
                "image_url": "https://example.com/images/p011.jpg",
                "is_active": True
            }
        ]
def clean_metadata(metadata):
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            if key == "rental_price":
                cleaned[key] = 0.0  # Default to 0.0 for null rental prices
            else:
                cleaned[key] = ""   # Default to empty string for other null values
        else:
            cleaned[key] = value
    return cleaned



# 🧠 Prepare vectors for Pinecone
vectors = []

for item in product_data:
    # ✨ Generate embedding for the product description with 1024 dimensions
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=item["description"],
        dimensions=1536
    )
    print(f"Embedding vector length for {item['id']}: {len(response.data[0].embedding)}")  # Add this line here
    metadata = {
        "tenant_id": item["tenant_id"],
        "sku": item["sku"],
        "product_name": item["product_name"],
        "category": item["category"],
        "fabric": item["fabric"],
        "color": item["color"],
        "size": item["size"],
        "price": item["price"],
        "stock": item["stock"],
        "description": item["description"],
        "rental_price": item["rental_price"],
        "is_rental": item["is_rental"],
        "available_stock": item["available_stock"],
        "image_url": item["image_url"],
        "is_active": item["is_active"]
    }
    cleaned_metadata = clean_metadata(metadata)
    # 📌 Add vector with product metadata
    vectors.append({
        "id": item["id"],
        "values": response.data[0].embedding,
        "metadata": cleaned_metadata
    })
index = pinecone.Index(INDEX_NAME)
# 🚀 Upload all vectors to Pinecone in the selected namespace
index.upsert(vectors=vectors, namespace=NAMESPACE)

print("✅ Product data uploaded successfully to Pinecone with 1536-dimensional embeddings!")
