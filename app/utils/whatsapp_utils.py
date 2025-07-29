import os
import httpx

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_API_URL = "https://graph.facebook.com/v19.0"

async def send_product_card(phone_number_id, to, product):
    url = f"{WHATSAPP_API_URL}/{phone_number_id}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "image",
        "image": {
            "link": product["image_url"],
            "caption": (
                f"*{product['name_en']}*\n"
                f"{product['color']} | {product['type']}\n"
                f"Price: ₹{product['price']}\n"
                "Reply with 'Order' or 'Rent' to proceed!"
            )
        }
    }
    async with httpx.AsyncClient() as client:
        await client.post(url, headers=headers, json=payload)
