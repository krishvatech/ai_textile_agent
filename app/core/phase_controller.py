# app/core/phase_controller.py
from app.core.session_memory import get_user_memory, set_user_memory
from app.db.models import Product, Order, Customer
from app.core.product_search import find_similar_products
from app.utils.whatsapp_utils import send_product_card, send_order_confirmation
from app.utils.voice_utils import tts_and_stream
from app.utils.payments import create_payment_link

PHASES = ["intro", "discovery", "delivery", "confirmation", "followup", "ending"]

async def handle_phase(user_id, channel, user_message, tenant_id, db, phone_number_id=None, to=None, call_sid=None):
    # 1. Get user session memory (multi-tenant, per-channel)
    session = await get_user_memory(user_id, tenant_id, channel)
    phase = session.get("phase", "intro")
    language = session.get("language", "en")

    # 2. Phase routing
    if phase == "intro":
        reply = "Welcome to our textile shop! What are you looking for today? (color/type/price)"
        session["phase"] = "discovery"
    elif phase == "discovery":
        # Use input classifier + product search
        criteria = extract_criteria_from_message(user_message)
        session.update(criteria)
        if all(k in session for k in ["color", "type"]):
            session["phase"] = "delivery"
            reply = None  # We'll show products below
        else:
            reply = "Can you tell me which color or type you want?"
    elif phase == "delivery":
        # Recommend matching products (use vector search)
        results = await find_similar_products(
            db, tenant_id,
            color=session.get("color"),
            product_type=session.get("type"),
            price_range=session.get("price_range"),
            top_k=3
        )
        if not results:
            reply = "Sorry, I couldn't find matching products. Would you like to try a different color or type?"
            session["phase"] = "discovery"
        else:
            # Send catalog (image+caption) via WhatsApp, or describe on voice
            if channel == "whatsapp":
                for product in results:
                    await send_product_card(phone_number_id, to, product)
                reply = "Here are some great matches! Want to order or rent?"
            else:
                reply = "I found these for you: " + "; ".join(
                    f"{p['name_en']} ({p['color']}) for ₹{p['price']}" for p in results
                ) + ". Would you like to order or rent any of these?"
            session["phase"] = "confirmation"
            session["product_suggestions"] = [p["id"] for p in results]
    elif phase == "confirmation":
        # Check for "order"/"rent" intent, create order, send payment link
        selection = extract_order_selection(user_message, session)
        if not selection:
            reply = "Which product would you like to order or rent? Please mention the name or number."
        else:
            product_id, order_type = selection
            customer_id = session.get("customer_id")  # Look up or create customer
            order = await create_order(
                db, tenant_id, customer_id, product_id, order_type
            )
            payment_link = await create_payment_link(order)
            session["order_id"] = order.id
            session["phase"] = "followup"
            if channel == "whatsapp":
                await send_order_confirmation(phone_number_id, to, order, order.product)
                reply = f"Please pay here to confirm: {payment_link}"
            else:
                reply = f"Please visit this link to pay and confirm your order: {payment_link}"
    elif phase == "followup":
        # After payment/order, collect feedback, offer more, or end session
        reply = "Thank you for your order! Would you like to share feedback or see new arrivals?"
        session["phase"] = "ending"
    elif phase == "ending":
        reply = "Thanks for chatting with us! Have a wonderful day!"
        session.clear()

    # Save session memory
    await set_user_memory(user_id, tenant_id, channel, session)

    # For voice, stream TTS; for WhatsApp, send text
    if channel == "voice" and call_sid:
        await tts_and_stream(call_sid, reply, language)
        return None
    else:
        return reply

# You need to implement extract_criteria_from_message, extract_order_selection, create_order
# These can use your AI intent classifier, NER, etc.
