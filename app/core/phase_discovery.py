from app.core.input_classifier import classify_input
from app.core.session_memory import get_user_memory, set_user_memory

async def handle_discovery(user_id, user_message, db, tenant_id):
    # Classify user intent (what are they looking for? color? type? price?)
    intent = classify_input(user_message)
    session = await get_user_memory(user_id)
    # Save preferences to session memory
    if intent.get("color"):
        session["preferred_color"] = intent["color"]
    if intent.get("product_type"):
        session["preferred_type"] = intent["product_type"]
    if intent.get("price_range"):
        session["preferred_price"] = intent["price_range"]
    await set_user_memory(user_id, session)
    # Return next phase or ready to recommend
    if all([session.get("preferred_color"), session.get("preferred_type")]):
        return "delivery", session
    return "discovery", session
