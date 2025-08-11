from dotenv import load_dotenv
import os
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import asyncio
from app.core.central_system_prompt import Textile_Prompt 
import random
from app.core.product_search import pinecone_fetch_records

load_dotenv()
api_key = os.getenv("GPT_API_KEY")
if not api_key:
    print("❌ Error: GPT_API_KEY not found in environment variables")
    exit(1)

client = AsyncOpenAI(api_key=api_key, timeout=6.0, max_retries=1)

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


async def generate_product_pitch_prompt(language: str, entities: Dict[str, Any], products: Optional[list] = None) -> str:
    lang_root = (language or "en-IN").split("-")[0].lower()
    lang_hint = {
        "en": "English (India) — use English words only. No Hindi, no Hinglish.",
        "hi": "Hindi in Devanagari script only. No English words or transliteration.",
        "gu": "Gujarati sentences in Gujarati script. Translate or transliterate product NAMES naturally into Gujarati (e.g., 'Georgette Party Saree' becomes 'જ્યોર્જેટ પાર્ટી સાડી'). No Hindi/Punjabi/English words except where necessary for clarity."
    }.get(lang_root, f"the exact locale {language}")
    sys_msg = (
        "You are a retail assistant for a textile shop. Your goal is to sound like a helpful human, not a robot reading a list. "
        "Write a very short, spoken pitch (1-2 natural sentences) for a voice assistant. "
        "Weave the provided product names into a conversational sentence. Do not invent facts. Obey language instructions exactly."
    )

    filtered_entities = {k: v for k, v in (entities or {}).items() if v not in [None, "", [], {}]}

    prompt = Textile_Prompt + (
        f"Reply ONLY in the exact locale given by {lang_hint} (same script, no transliteration). "
        "If products exist, mention all of the given product NAMES (exactly as provided); "
        "otherwise, compose a single-sku pitch from these collected entities: " + str(filtered_entities) + ". "
        f"Products: {products if products else None}. "
        "Keep it under ~60-80 words total. "
    )
    completion = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content.strip()


async def FollowUP_Question(
    intent_type: str,
    entities: Dict[str, Any],
    language: Optional[str] = "en-IN",
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

    lang_root = (language or "en-IN").split("-")[0].lower()
    lang_hint = {
        "en": "English (India) — English only, no Hindi/Hinglish",
        "hi": "Hindi in Devanagari script — no English/Hinglish",
        "gu": "Gujarati script — no Hindi/English",
    }.get(lang_root, f"the exact locale {language}")
    
    # Recent session for context (optional, helps GPT personalize)
    session_text = ""
    if session_history:
        relevant_history = session_history[-5:]
        conv_lines = [f"{m['role'].capitalize()}: {m['content']}" for m in relevant_history]
        session_text = "Conversation so far:\n" + "\n".join(conv_lines) + "\n"

    # Prompt instructing GPT to only ask about these N fields
    prompt = Textile_Prompt + (
        f"{session_text}"
        f"Collected details so far: { {k: v for k, v in entities.items() if v} }\n"
        f"Still missing: {merged_fields}.\n"
        f"Ask naturally and politely for ONLY these, like 'Would you like to rent or buy? Any preferred price or fabric?'\n"
        f"Do not mention any other fields. Keep it very brief. "
        f"Reply in {language.upper()}. Only output the question."
        f"Write in {lang_hint}. Output only one question."

    )
    completion = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert, concise, friendly assistant. Respect language instructions strictly."},
            {"role": "user", "content": prompt}
        ]
        # gpt-5-mini: don't pass temperature/max_tokens
    )
    return completion.choices[0].message.content.strip()


def normalize_entities(entities):
    new_entities = {}
    for k, v in entities.items():
        if isinstance(v, str):
            new_entities[k] = v.lower().replace(" ", "")
        else:
            new_entities[k] = v
    return new_entities

async def generate_greeting_reply(language, session_history=None,mode: str = "call") -> str:
    # More concise, shop-aware greeting 
    emoji_instruction = "Do not use emojis." if mode == "call" else "Use emojis if you like."
    prompt = Textile_Prompt + (
        f"{'Recent conversation: ' + str(session_history) if session_history else ''}\n"
        f"Greet the customer in a warm, short (1-2 sentences), conversational way in {language.upper()}. "
        f"If this is the first message, welcome them to our shop. "
        f"If ongoing, give a friendly brief follow-up. {emoji_instruction} "
        f"Only output the greeting, no product suggestions."
    )
    try:
        completion = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an expert conversation starter and friendly textile assistant."},
                {"role": "user", "content": prompt}
            ]
            # gpt-5-mini: don't pass temperature/max_tokens
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


def clean_entities_for_pinecone(entities):
    return {k:v for k,v in entities.items() if v not in [None, '', [], {}]}

async def analyze_message(text: str, tenant_id: int,language: str = "en-US",intent: str | None = None,new_entities: dict | None = None,intent_confidence: float = 0.0,mode: str = "call") -> Dict[str, Any]:
    language = language
    logging.info(f"Detected language in analyze_message: {language}")
    intent_type=intent
    logging.info(f"Detected intent_type in analyze_message: {intent_type}")
    entities=new_entities
    logging.info(f"Detected entities in analyze_message: {entities}")
    confidence=intent_confidence
    logging.info(f"Detected confidence in analyze_message: {confidence}")
    tenant_id=tenant_id
    logging.info(f"Tenant id ==== {tenant_id}======")
    mode=mode
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
        reply = await generate_greeting_reply(language, session_history=history,mode=mode)
        history.append({"role": "assistant", "content": reply})
        session_memory[tenant_id] = history
        return {
            "input_text": text,
            "language": language,
            "intent_type": intent_type,
            "reply_text": reply,
            "history": history,
            "collected_entities": acc_entities
        }
    elif intent_type == "product_search":
        filtered_entities = filter_non_empty_entities(acc_entities)
        filtered_entities_norm = normalize_entities(filtered_entities)
        # ADD THIS LINE:
        filtered_entities_norm = clean_entities_for_pinecone(filtered_entities_norm)
        pinecone_data = await pinecone_fetch_records(filtered_entities_norm, tenant_id)
        # --- Format product results and extra info (improved) ---
        product_lines = []
        for product in (pinecone_data or []):
            name = product.get("product_name") or "Unnamed Product"

            # Treat is_rental explicitly; don't default to truthy
            is_rental = product.get("is_rental")
            availability = "Available For Rent" if is_rental is True else "Available For Sale"

            # Optional attributes
            fabric = (product.get("fabric") or "").strip()
            color = (product.get("color") or "").strip()
            size = (product.get("size") or "").strip()

            # Prices
            rental_price = product.get("rental_price")
            price = product.get("price")

            details = [availability]

            # Meta details
            meta_bits = []
            if fabric:
                meta_bits.append(f"{fabric}")
            if color:
                meta_bits.append(f"{color}")
            if size:
                meta_bits.append(f"{size}")
            if meta_bits:
                details.append(" | ".join(meta_bits))

            # Pricing details (adjust currency as needed)
            price_bits = []
            if is_rental is True and rental_price not in [None, "", 0]:
                price_bits.append(f"Rent: ₹{rental_price}")
            if (is_rental is False or is_rental is None) and price not in [None, "", 0]:
                price_bits.append(f"Price: ₹{price}")
            if price_bits:
                details.append(" | ".join(price_bits))

            # Final formatted line
            line = f"{name} — {' • '.join(details)}"
            product_lines.append(line)

        # Build the products text
        if product_lines:
            category = filtered_entities.get("color") or filtered_entities.get("category") or "products"
            products_text = f"Here are our {category}:\n" + "\n".join(product_lines)
        else:
            products_text = "Sorry, no products match your search so far."

        
        followup = await FollowUP_Question(intent_type, acc_entities, language, session_history=history)
        # Final consolidated reply string for this turn
        reply_text = f"{products_text}\n{followup}"
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
                "reply_text": reply_text
            }
        elif mode== "chat":
            history.append({"role": "assistant", "content": reply_text})
            session_memory[tenant_id] = history
            print("="*20)
            print(reply_text)
            print("="*20)
            return {
                "pinecone_data": pinecone_data,
                "intent_type": intent_type,
                "language": language,
                "tenant_id": tenant_id,
                "history": history,
                "collected_entities": acc_entities,
                "followup_reply": followup,
                "reply_text": reply_text
            }
    else:
        return "<--> Upcoming Modules Are Under Development <-->"