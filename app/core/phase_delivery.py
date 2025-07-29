from app.core.semantic_similarity import find_similar_products
from app.core.session_memory import get_user_memory

async def handle_delivery(user_id, db, tenant_id):
    session = await get_user_memory(user_id)
    # Use semantic similarity (your product embedding code) to find matches
    results = await find_similar_products(
        db=db,
        tenant_id=tenant_id,
        color=session.get("preferred_color"),
        product_type=session.get("preferred_type"),
        price_range=session.get("preferred_price"),
        top_k=3
    )
    # Prepare WhatsApp image+caption messages using whatsapp.py utils
    messages = []
    for product in results:
        msg = {
            "image_url": product["image_url"],
            "caption": f"{product['name']} – {product['color']} – ₹{product['price']}\nWant to book/rent?"
        }
        messages.append(msg)
    return messages
