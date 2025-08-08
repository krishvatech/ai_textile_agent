from app.core.product_search import pinecone_fetch_records
from app.core.ai_reply import analyze_message,generate_greeting_reply
from app.core.session_manager import SessionManager
from app.core.lang_utils import detect_language

async def handle_user_message(user_id, message, tenant_id, shop_name, db, channel="whatsapp"):
    session = await SessionManager.get_session(user_id)
    lang = session.get("lang") or detect_language(message)
    session["lang"] = lang

    if session.get("pending_action") == "order":
        session["address"] = message
        reply_text = await generate_greeting_reply(message, [], shop_name, action="order", language=lang)
        await SessionManager.clear_session(user_id)
        return reply_text, [], session

    results = pinecone_fetch_records(message, tenant_id)
    # Fetch products from DB (pseudo-code):
    # products = [await db.get(Product, r["product_id"]) for r in results]
    # Here, products must be dicts with name, price, color, image_url
    products = [] # Fill from DB in your code

    if any(w in message.lower() for w in ["buy", "order", "purchase"]):
        session["pending_action"] = "order"
        reply_text = await analyze_message(message, products, shop_name, action="order", language=lang)
    elif any(w in message.lower() for w in ["rent", "rental", "book"]):
        session["pending_action"] = "rental"
        reply_text = await analyze_message(message, products, shop_name, action="rental", language=lang)
    else:
        reply_text = await analyze_message(message, products, shop_name, language=lang)
        session["last_products"] = products

    await SessionManager.set_session(user_id, session)
    images = [(p["image_url"], f"{i+1}. {p['name']} – ₹{p['price']} ({p['color']})") for i, p in enumerate(products) if p.get("image_url")]
    return reply_text, images, session
