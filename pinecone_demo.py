from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()

# ✅ Set OpenAI & Pinecone API keys
client = OpenAI()
pc = Pinecone(api_key='pcsk_6vwgPk_KJYut1gc9HRb3PziHXjPAeMfonV8rxauXFnhgRqztm4k2tQVfyEY8ftucbJqq5i')  # Replace with your actual Pinecone key

# ✅ Connect to your Pinecone index
index = pc.Index('ai-agent-textile')  # Index for AI agent in textile industry
namespace = 'ns1'  # Change if you're using a different namespace like "ns2"

product_data = [
    {"id": "P001", "product_name": "Banarasi Silk Saree", "category": "Saree", "fabric": "Silk", "color": "Red-Gold", "size": "Free Size", "price": 2999, "stock": "Available", "description": "Elegant Banarasi silk saree with zari work."},
    {"id": "P002", "product_name": "Chiffon Party Wear Saree", "category": "Saree", "fabric": "Chiffon", "color": "Blue", "size": "Free Size", "price": 1599, "stock": "Available", "description": "Lightweight blue chiffon saree ideal for parties."},
    {"id": "P003", "product_name": "Cotton Daily Wear Kurti", "category": "Kurti", "fabric": "Cotton", "color": "White", "size": "M, L, XL", "price": 499, "stock": "Available", "description": "Comfortable white kurti for everyday wear."},
    {"id": "P004", "product_name": "Rayon Printed Kurti", "category": "Kurti", "fabric": "Rayon", "color": "Yellow", "size": "S, M, L", "price": 599, "stock": "Available", "description": "Yellow rayon kurti with floral prints."},
    {"id": "P005", "product_name": "Designer Lehenga Choli", "category": "Lehenga", "fabric": "Net", "color": "Pink", "size": "Free Size", "price": 4999, "stock": "Available", "description": "Pink designer lehenga with detailed embroidery."},
    {"id": "P006", "product_name": "Men's Cotton Shirt", "category": "Shirt", "fabric": "Cotton", "color": "Light Blue", "size": "M, L, XL, XXL", "price": 799, "stock": "Available", "description": "Light blue shirt perfect for summer."},
    {"id": "P007", "product_name": "Georgette Anarkali Suit", "category": "Suit", "fabric": "Georgette", "color": "Green", "size": "M, L, XL", "price": 1799, "stock": "Available", "description": "Flowy green georgette anarkali suit."},
    {"id": "P008", "product_name": "Casual Palazzo Pants", "category": "Palazzo", "fabric": "Cotton", "color": "Black", "size": "S, M, L, XL", "price": 399, "stock": "Available", "description": "Stretchy cotton palazzo for daily wear."},
    {"id": "P009", "product_name": "Formal Men's Trousers", "category": "Trousers", "fabric": "Polyester", "color": "Grey", "size": "32, 34, 36, 38", "price": 999, "stock": "Available", "description": "Grey formal trousers for office use."},
    {"id": "P010", "product_name": "Girls Frock Dress", "category": "Dress", "fabric": "Net", "color": "Peach", "size": "2-6 Years", "price": 599, "stock": "Available", "description": "Peach net frock for girls, party wear."},
    {"id": "P011", "product_name": "Men's Denim Jeans", "category": "Jeans", "fabric": "Denim", "color": "Blue", "size": "30, 32, 34, 36", "price": 1299, "stock": "Available", "description": "Classic blue denim jeans for men."},
    {"id": "P012", "product_name": "Women's Printed Skirt", "category": "Skirt", "fabric": "Cotton", "color": "Multi", "size": "S, M, L", "price": 749, "stock": "Available", "description": "Colorful printed cotton skirt."},
    {"id": "P013", "product_name": "Silk Dupatta", "category": "Dupatta", "fabric": "Silk", "color": "Maroon", "size": "2.5m", "price": 399, "stock": "Available", "description": "Soft silk maroon dupatta."},
    {"id": "P014", "product_name": "Kids Cotton Shorts", "category": "Shorts", "fabric": "Cotton", "color": "Navy", "size": "2-10 Years", "price": 299, "stock": "Available", "description": "Comfortable navy shorts for kids."},
    {"id": "P015", "product_name": "Woolen Winter Shawl", "category": "Shawl", "fabric": "Wool", "color": "Brown", "size": "Free Size", "price": 899, "stock": "Available", "description": "Brown woolen shawl for winter."},
    {"id": "P016", "product_name": "Chanderi Silk Dupatta", "category": "Dupatta", "fabric": "Chanderi Silk", "color": "Off White", "size": "2.3m", "price": 549, "stock": "Available", "description": "Elegant Chanderi silk dupatta."},
    {"id": "P017", "product_name": "Fancy Lycra Leggings", "category": "Leggings", "fabric": "Lycra", "color": "Pink", "size": "S, M, L, XL", "price": 349, "stock": "Available", "description": "Pink lycra leggings, stretchable."},
    {"id": "P018", "product_name": "Embroidered Kurta Set", "category": "Kurta Set", "fabric": "Cotton Blend", "color": "Grey", "size": "M, L, XL", "price": 999, "stock": "Available", "description": "Grey embroidered kurta set with pants."},
    {"id": "P019", "product_name": "Printed Night Suit", "category": "Nightwear", "fabric": "Cotton", "color": "White-Blue", "size": "M, L, XL", "price": 649, "stock": "Available", "description": "Soft printed night suit set."},
    {"id": "P020", "product_name": "Boys Casual T-Shirt", "category": "T-Shirt", "fabric": "Cotton", "color": "Green", "size": "4-12 Years", "price": 349, "stock": "Available", "description": "Cool green t-shirt for boys."},
    {"id": "P021", "product_name": "Men's Kurta Pajama", "category": "Kurta Pajama", "fabric": "Cotton", "color": "Beige", "size": "M, L, XL, XXL", "price": 1099, "stock": "Available", "description": "Traditional beige kurta pajama."},
    {"id": "P022", "product_name": "Women’s Long Shrug", "category": "Shrug", "fabric": "Rayon", "color": "Black", "size": "S, M, L", "price": 499, "stock": "Available", "description": "Stylish black rayon shrug."},
    {"id": "P023", "product_name": "Girls Embroidered Top", "category": "Top", "fabric": "Cotton", "color": "Yellow", "size": "8-14 Years", "price": 399, "stock": "Available", "description": "Bright yellow top with embroidery."},
    {"id": "P024", "product_name": "Cotton Saree with Blouse", "category": "Saree", "fabric": "Cotton", "color": "Sky Blue", "size": "Free Size", "price": 999, "stock": "Available", "description": "Sky blue cotton saree with matching blouse."},
    {"id": "P025", "product_name": "Velvet Party Blazer", "category": "Blazer", "fabric": "Velvet", "color": "Dark Blue", "size": "M, L, XL", "price": 2299, "stock": "Available", "description": "Dark blue velvet blazer for parties."},
    {"id": "P026", "product_name": "Boys Denim Jacket", "category": "Jacket", "fabric": "Denim", "color": "Blue", "size": "6-16 Years", "price": 1199, "stock": "Available", "description": "Trendy blue denim jacket for boys."},
    {"id": "P027", "product_name": "Printed Rayon Gown", "category": "Gown", "fabric": "Rayon", "color": "Purple", "size": "M, L, XL", "price": 1499, "stock": "Available", "description": "Purple rayon gown with prints."},
    {"id": "P028", "product_name": "Men's Woolen Sweater", "category": "Sweater", "fabric": "Wool", "color": "Grey", "size": "M, L, XL, XXL", "price": 899, "stock": "Available", "description": "Warm grey woolen sweater for men."},
    {"id": "P029", "product_name": "Girls Net Party Dress", "category": "Dress", "fabric": "Net", "color": "Pink", "size": "3-8 Years", "price": 799, "stock": "Available", "description": "Pink net party dress for girls."},
    {"id": "P030", "product_name": "Men's Formal Suit", "category": "Suit", "fabric": "Polyester", "color": "Black", "size": "M, L, XL, XXL", "price": 3999, "stock": "Available", "description": "Classic black suit for formal occasions."}
]

# 🧠 Prepare vectors for Pinecone
vectors = []

for item in product_data:
    # ✨ Generate embedding for the product description with 1024 dimensions
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=item["description"],
    )
    print(f"Embedding vector length for {item['id']}: {len(response.data[0].embedding)}")  # Add this line here

    # 📌 Add vector with product metadata
    vectors.append({
        "id": item["id"],
        "values": response.data[0].embedding,
        "metadata": {
            "product_name": item["product_name"],
            "category": item["category"],
            "fabric": item["fabric"],
            "color": item["color"],
            "size": item["size"],
            "price": item["price"],
            "stock": item["stock"],
            "description": item["description"]
        }
    })

# 🚀 Upload all vectors to Pinecone in the selected namespace
index.upsert(vectors=vectors, namespace=namespace)

print("✅ Product data uploaded successfully to Pinecone with 1024-dimensional embeddings!")

