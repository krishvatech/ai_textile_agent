from dotenv import load_dotenv
import os
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import asyncio
import random
from app.core.lang_utils import detect_language
from app.core.intent_utils import detect_textile_intent_openai

load_dotenv()
api_key = os.getenv("GPT_API_KEY")
if not api_key:
    print("❌ Error: GPT_API_KEY not found in environment variables")
    exit(1)

# --- In-memory session stores
session_memory: Dict[Any, List[Dict[str, str]]] = {}  # Conversation history
session_entities: Dict[Any, Dict[str, Any]] = {}       # Merged entities per session
last_main_intent_by_session: Dict[Any, str] = {}       # Remember last main intent



MAIN_INTENTS = {
    "product_search", "catalog_request", "order_placement", "order_status",
    "price_inquiry", "discount_inquiry", "availability_check"
}
REFINEMENT_INTENTS = {
    "rental_inquiry", "color_preference", "size_query", "fabric_inquiry",
    "delivery_inquiry", "payment_query"
}

def filter_non_empty_entities(entities: dict) -> dict:
    """
    Returns a dict of only non-empty (not None, '', [], or {}) entity fields.
    """
    return {k: v for k, v in entities.items()
            if v not in [None, '', [], {}]}

async def pinecone_fetch_records(entities: Dict) -> str:
    # Replace this with your real Pinecone search logic!
    return "Similar data from Pinecone based on collected entities: " + str(entities)

async def generate_product_pitch_prompt(language: str, entities: Dict[str, Any], products: Optional[list] = None) -> str:
    """
    Use GPT to produce a short, natural, TTS-friendly pitch in the caller's language.
    - language: BCP-47 like 'gu-IN', 'hi-IN', 'kn-IN', 'en-IN'
    - entities: merged collected entities
    - products: optional list[dict] with keys like name/category/color/fabric/size/price/rental_price
    """
    sys_msg = (
        "You are a retail assistant for a textile shop. "
        "Write a very short spoken pitch (max 2 sentences, no emojis, no numbering, no bullets). "
        "Natural tone, suitable for voice. Do not invent facts. "
        f"Respond EXCLUSIVELY in the language of the locale {language} (e.g., Gujarati for 'gu-IN', no mixing with Hindi or English)."
    )

    # Filter non-empty entities directly
    filtered_entities = {k: v for k, v in (entities or {}).items() if v not in [None, "", [], {}]}

    # Build prompt with integrated context
    prompt = (
        f"Reply ONLY in the exact locale given by {language} (same script, no transliteration). "
        "If products exist, briefly pitch up to 3 options (color/fabric/category, size if present, and price or rental price); "
        "otherwise, compose a single-sku pitch from these collected entities: " + str(filtered_entities) + ". "
        f"Products: {products[:3] if products else None}. "
        "Keep it under ~30 words total. "
    )

    client = AsyncOpenAI(api_key=api_key)
    completion = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=90,
    )
    return completion.choices[0].message.content.strip()


async def FollowUP_Question(
    intent_type: str,
    entities: Dict[str, Any],
    language: Optional[str] = "en",
    session_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Generates a short, merged follow-up question asking for only the top 2-3 missing entities.
    """
    # Find missing fields (consider empty as missing)
    missing_fields = [k for k, v in entities.items()
                      if v is None or v == "" or (isinstance(v, (list, dict)) and not v)]
    if not missing_fields:
        return "Thank you. I have all the information I need for your request!"

    # Priority for asking
    entity_priority = [
        "is_rental", "rental_price", "price", "fabric",
        "size", "color", "category", "product_name",
        "quantity", "location", "occasion", "type"
    ]
    field_display_names = {
        "is_rental": "rental",
        "rental_price": "rental price",
        "price": "price",
        "fabric": "fabric",
        "size": "size",
        "color": "color",
        "category": "category",
        "product_name": "product",
        "quantity": "quantity",
        "location": "location",
        "occasion": "occasion",
        "type": "gender/type"
    }
    # Sort and select only top 2 or 3 missing fields
    missing_sorted = sorted(missing_fields, key=lambda x: entity_priority.index(x) if x in entity_priority else 999)
    max_fields = 3
    missing_short = missing_sorted[:max_fields]
    merged_fields = ", ".join([field_display_names.get(f, f) for f in missing_short])

    # Recent session for context (optional, helps GPT personalize)
    session_text = ""
    if session_history:
        relevant_history = session_history[-5:]
        conv_lines = [f"{m['role'].capitalize()}: {m['content']}" for m in relevant_history]
        session_text = "Conversation so far:\n" + "\n".join(conv_lines) + "\n"

    # Prompt instructing GPT to only ask about these N fields
    prompt = (
        f"You are a friendly WhatsApp assistant for a textile and clothing shop.\n"
        f"{session_text}"
        f"Collected details so far: { {k: v for k, v in entities.items() if v} }\n"
        f"Still missing: {merged_fields}.\n"
        f"Ask for ONLY these {len(missing_short)} details in a single, short, conversational question, e.g., 'Want to filter by rental or fabric?' "
        f"Do not mention any other fields. Keep it very brief. "
        f"Reply in {language.upper()}. Only output the question."
        f"Reply EXCLUSIVELY in the locale {language.upper()} (no mixing with other languages like Hindi). Only output the question."
    )

    client = AsyncOpenAI(api_key=api_key)
    completion = await client.chat.completions.create(
        model="gpt-4.1-mini",  # Or your available model
        messages=[
            {"role": "system", "content": "You are an expert, concise, friendly assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=70
    )
    questions = completion.choices[0].message.content.strip()
    return questions

async def generate_greeting_reply(language, session_history=None) -> str:
    # More concise, shop-aware greeting prompt
    prompt = (
        f"You are a friendly WhatsApp assistant for our textile and clothing business.\n"
        f"{'Recent conversation: ' + str(session_history) if session_history else ''}\n"
        f"Greet the customer in a warm, short (1-2 sentences), conversational way in {language.upper()}. "
        f"If this is the first message, welcome them to our shop. "
        f"If ongoing, give a friendly brief follow-up. Use emojis if you like. "
        f"Only output the greeting, no product suggestions."
    )
    try:
        client = AsyncOpenAI(api_key=api_key)
        completion = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an expert conversation starter and friendly textile assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=60
        )
        reply = completion.choices[0].message.content.strip()
        if reply:
            return reply
    except Exception as e:
        print("GPT error, falling back to random greeting. (Error:", str(e), ")")
    greetings = [
        "Hello! 👋 Welcome to our textile shop.",
        "Hi there! 😊 How can I help you with fabrics or clothing?",
        "Welcome! Let me know what you're searching for today.",
        "Hello! What are you looking for in clothing or textiles?"
    ]
    return random.choice(greetings)

async def analyze_message(text: str, tenant_id=None,language: str = "en-US",intent: str | None = None,new_entities: dict | None = None,intent_confidence: float = 0.0,mode: str = "call") -> Dict[str, Any]:
    language = language
    logging.info(f"Detected language in analyze_message: {language}")
    intent_type=intent
    logging.info(f"Detected intent_type in analyze_message: {intent_type}")
    entities=new_entities
    logging.info(f"Detected entities in analyze_message: {entities}")
    confidence=intent_confidence
    logging.info(f"Detected confidence in analyze_message: {confidence}")
    
    history = session_memory.get(tenant_id, [])
    acc_entities = session_entities.get(tenant_id, None)
    last_main_intent = last_main_intent_by_session.get(tenant_id, None)
    history.append({"role": "user", "content": text})
    # --- Merge entities
    if acc_entities is None:
        acc_entities = entities.copy()
    else:
        for k, v in entities.items():
            if v not in [None, '', [], {}]:
                acc_entities[k] = v

    # --- Rental/Purchase logic
    is_rental = acc_entities.get("is_rental")

    if is_rental is True:
        acc_entities.pop("price", None)        # Remove 'price' if present
    elif is_rental is False:
        acc_entities.pop("rental_price", None) # Remove 'rental_price' if present


    session_entities[tenant_id] = acc_entities

    # === INTENT STICKY LOGIC ===
    if intent_type in REFINEMENT_INTENTS and last_main_intent:
        intent_type = last_main_intent  # Don't leave main flow for refinement
    if intent_type in MAIN_INTENTS:
        last_main_intent_by_session[tenant_id] = intent_type

    # --- Respond
    if intent_type == "greeting":
        reply = await generate_greeting_reply(language, session_history=history)
        history.append({"role": "assistant", "content": reply})
        session_memory[tenant_id] = history
        return {
            "input_text": text,
            "language": language,
            "intent_type": intent_type,
            "followup_reply": reply,
            "history": history,
            "collected_entities": acc_entities
        }
    elif intent_type == "product_search":
        filtered_entities = filter_non_empty_entities(acc_entities)
        pinecone_data = await pinecone_fetch_records(filtered_entities)
        followup = await FollowUP_Question(intent_type, acc_entities, language, session_history=history)
        if mode == "call":
            # Generate and speak full response
            spoken_pitch = await generate_product_pitch_prompt(language, acc_entities, pinecone_data)
            voice_response = f"{spoken_pitch} {followup}"
            history.append({"role": "assistant", "content": voice_response})
            session_memory[tenant_id] = history
            return {
                "pinecone_data": pinecone_data,
                "intent_type": intent_type,
                "language": language,
                "tenant_id": tenant_id,
                "history": history,
                "collected_entities": acc_entities,
                "answer": voice_response,  # For TTS in call
                "followup_reply": followup,
            }
        elif mode== "chat":
            history.append({"role": "assistant", "content": followup})
            session_memory[tenant_id] = history
            return {
                "pinecone_data": pinecone_data,
                "intent_type": intent_type,
                "language": language,
                "tenant_id": tenant_id,
                "history": history,
                "collected_entities": acc_entities,
                "followup_reply": followup,
            }
    else:
        return "<--> Upcoming Modules Are Under Development <-->"
    
async def main():
    # Sample test data
    test_tenant_id = "test_user_123"
    test_text = "I'm looking for a red shirt"
    test_language = "gu-IN"  # English (India); change to 'as-IN' for Assamese testing
    test_intent = "product_search"
    test_entities = {
        "color": "red",
        "category": "shirt",
        "size": None,  # Intentionally missing for follow-up testing
        "price": None
    }
    test_confidence = 0.85
    test_mode = "call"  # Test "chat" or "call" here

    try:
        # Simulate analysis
        result = await analyze_message(
            text=test_text,
            tenant_id=test_tenant_id,
            language=test_language,
            intent=test_intent,
            new_entities=test_entities,
            intent_confidence=test_confidence,
            mode=test_mode  # Added for mode-specific testing
        )

        # Print results for inspection
        print("\n--- Local Test Results ---")
        print(f"Mode: {test_mode}")
        print(f"Intent Type: {result.get('intent_type')}")
        print(f"Collected Entities: {result.get('collected_entities')}")
        print(f"Pinecone Data: {result.get('pinecone_data')}")
        print(f"Spoken Pitch (Answer): {result.get('answer')}")
        print(f"Follow-up Reply: {result.get('followup_reply')}")
        print(f"History: {result.get('history')}")
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())