from fastapi import APIRouter, Request, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_db
from app.db.models import Tenant, Product, Customer, Order
from app.core.product_search import search_products
from app.core.ai_reply import generate_reply
from app.core.session_memory import get_session, set_session
import os, httpx
from datetime import datetime

router = APIRouter()

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_API_URL = "https://graph.facebook.com/v19.0"

async def send_whatsapp_message(phone_number_id, to, text):
    url = f"{WHATSAPP_API_URL}/{phone_number_id}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}}
    async with httpx.AsyncClient() as client:
        await client.post(url, headers=headers, json=payload)

async def send_whatsapp_image(phone_number_id, to, image_url, caption):
    url = f"{WHATSAPP_API_URL}/{phone_number_id}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to, "type": "image", "image": {"link": image_url, "caption": caption}}
    async with httpx.AsyncClient() as client:
        await client.post(url, headers=headers, json=payload)

@router.post("/webhook")
async def whatsapp_webhook(
    req: Request, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)
):
    body = await req.json()
    try:
        entry = body["entry"][0]["changes"][0]["value"]
        message = entry.get("messages", [])[0]
        wa_to_number = entry["metadata"]["display_phone_number"]
        wa_from_number = message["from"]
        user_msg = message.get("text", {}).get("body", "")
        phone_number_id = entry["metadata"]["phone_number_id"]

        result = await db.execute(
            "SELECT * FROM tenants WHERE whatsapp_number = :number", {"number": wa_to_number}
        )
        tenant = result.fetchone()
        if not tenant:
            return {"error": "No tenant for this WhatsApp number."}

        user_id = wa_from_number
        session = get_session(user_id)

        # --- Order/rental flow ---
        if session.get("pending_action") == "order":
            if "address" not in session:
                set_session(user_id, {"address": user_msg})
                reply_text = "Thanks! Please confirm, do you want to place the order now? (yes/no)"
                background_tasks.add_task(send_whatsapp_message, phone_number_id, wa_from_number, reply_text)
                return {"status": "continue"}
            elif user_msg.strip().lower() in ["yes", "confirm", "ok"]:
                customer = await db.execute(
                    "SELECT * FROM customers WHERE whatsapp_id=:w AND tenant_id=:tid",
                    {"w": wa_from_number, "tid": tenant.id}
                )
                cust = customer.fetchone()
                if not cust:
                    cust = Customer(whatsapp_id=wa_from_number, tenant_id=tenant.id)
                    db.add(cust)
                    await db.commit()
                    await db.refresh(cust)
                order = Order(
                    tenant_id=tenant.id,
                    customer_id=cust.id,
                    product_id=session["product_id"],
                    order_type="purchase",
                    price=session["price"],
                    created_at=datetime.utcnow(),
                )
                db.add(order)
                await db.commit()
                set_session(user_id, {"pending_action": None})
                reply_text = "Order placed! Thank you 😊."
                background_tasks.add_task(send_whatsapp_message, phone_number_id, wa_from_number, reply_text)
                return {"status": "ordered"}

        # --- New order intent ---
        if any(kw in user_msg.lower() for kw in ["buy", "order", "purchase"]):
            if session.get("last_products"):
                idx = 0
                for word in user_msg.split():
                    if word.isdigit():
                        idx = int(word) - 1
                        break
                try:
                    product = session["last_products"][idx]
                    set_session(user_id, {
                        "pending_action": "order",
                        "product_id": product["id"],
                        "price": product["price"]
                    })
                    reply_text = f"Great! Please provide your delivery address to complete your order for '{product['name']}' (₹{product['price']})."
                    background_tasks.add_task(send_whatsapp_message, phone_number_id, wa_from_number, reply_text)
                    return {"status": "start_order"}
                except Exception:
                    pass
            reply_text = "Please tell me which product you'd like to order."
            background_tasks.add_task(send_whatsapp_message, phone_number_id, wa_from_number, reply_text)
            return {"status": "ask_order"}

        # --- Product search ---
        results = await search_products(user_msg, tenant.id, top_k=3)
        products = []
        for r in results:
            prod = await db.get(Product, r["product_id"])
            products.append({"id": prod.id, "name": prod.name, "price": prod.price, "color": prod.color, "image_url": prod.image_url})

        reply_text = generate_reply(user_msg, products, tenant.name)
        if products:
            set_session(user_id, {"last_products": products})
            reply_text += "\n\nTo order, reply 'order 1' or 'buy 2'."
            for i, prod in enumerate(products):
                if prod["image_url"]:
                    caption = f"{i+1}. {prod['name']} – ₹{prod['price']} ({prod['color']})"
                    background_tasks.add_task(
                        send_whatsapp_image, phone_number_id, wa_from_number, prod["image_url"], caption
                    )

        background_tasks.add_task(send_whatsapp_message, phone_number_id, wa_from_number, reply_text)
        return {"status": "replied"}

    except Exception as e:
        print("Webhook parsing error:", e)
        return {"error": str(e)}
