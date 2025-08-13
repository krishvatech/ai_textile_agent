from dotenv import load_dotenv
import os
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from dateutil import parser as dateparser
from app.core.central_system_prompt import Textile_Prompt 
import random
from sqlalchemy import text  # Import if not already present
from app.db.session import SessionLocal
from app.core.rental_utils import is_variant_available
from app.core.product_search import pinecone_fetch_records

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
    if not isinstance(entities, dict):
        logging.warning(f"filter_non_empty_entities: Input is not a dict ({type(entities)}), returning empty.")
        return {}
    return {k: v for k, v in entities.items()
            if v not in [None, '', [], {}]}


async def generate_product_pitch_prompt(language: str, entities: Dict[str, Any], products: Optional[list] = None) -> str:
    """
    Use GPT to produce a short, natural, TTS-friendly pitch in the caller's language.
    - language: BCP-47 like 'gu-IN', 'hi-IN', 'kn-IN', 'en-IN'
    - entities: merged collected entities
    - products: optional list[dict] with keys like name/category/color/fabric/size/price/rental_price
    """
    if not isinstance(entities, dict):
        logging.warning(f"generate_product_pitch_prompt: entities is not a dict ({type(entities)}), using empty.")
        entities = {}
    lang_root = (language or "en-IN").split("-")[0].lower()
    lang_hint = {
        "en": "English (India) — use English words only. No Hindi, no Hinglish.",
        "hi": "Hindi in Devanagari script only. No English words or transliteration.",
        "gu": "Gujarati sentences, but keep product NAMES exactly as provided (no translation or transliteration of names). No Hindi/Punjabi/English words except the product names."
    }.get(lang_root, f"the exact locale {language}")
    sys_msg = (
        "You are a retail assistant for a textile shop. "
        "Write a very short spoken pitch (max 2 sentences). Natural tone, for voice. "
        "Mention every provided product name exactly once, without changing, translating, or transliterating it. "
        "Do not invent facts. Obey language instructions exactly."
    )

    # Filter non-empty entities directly
    filtered_entities = {k: v for k, v in (entities or {}).items() if v not in [None, "", [], {}]}

    # Build prompt with integrated context
    prompt = Textile_Prompt + (
        f"Reply ONLY in the exact locale given by {lang_hint} (same script, no transliteration). "
        "If products exist, mention all of the given product NAMES (exactly as provided); "
        "otherwise, compose a single-sku pitch from these collected entities: " + str(filtered_entities) + ". "
        f"Products: {products if products else None}. "
        "Keep it under ~60-80 words total. "
    )

    client = AsyncOpenAI(api_key=api_key)
    completion = await client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ],
        # gpt-5-mini: do not send temperature/max_tokens
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

    entity_priority = [
        "is_rental","occasion", "fabric",
        "size", "color", "category", "product_name",
        "quantity", "location","type","price","rental_price",
    ]
    field_display_names = {
        "is_rental": "rental",
        "occasion": "occasion",
        "fabric": "fabric",
        "size": "size",
        "color": "color",
        "category": "category",
        "product_name": "product",
        "quantity": "quantity",
        "location": "location",
        "type": "gender/type",
        "price": "price",
        "rental_price": "rental price",
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
        f"You are a friendly assistant for a textile and clothing shop.\n"
        f"{session_text}"
        f"Collected details so far: { {k: v for k, v in entities.items() if v} }\n"
        f"Still missing: {merged_fields}.\n"
        f"Ask naturally and politely for ONLY these, like 'Would you like to rent or buy? Any preferred price or fabric?'\n"
        f"Do not mention any other fields. Keep it very brief. "
        f"Reply in {language.upper()}. Only output the question."
        f"Write in {lang_hint}. Output only one question."

    )

    client = AsyncOpenAI(api_key=api_key)
    completion = await client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are an expert, concise, friendly assistant. Respect language instructions strictly."},
            {"role": "user", "content": prompt}
        ]
        # gpt-5-mini: don't pass temperature/max_tokens
    )
    return completion.choices[0].message.content.strip()


def normalize_entities(entities):
    new_entities = {}
    # Keys where we preserve spaces (add more if needed, e.g., 'size', 'occasion')
    preserve_space_keys = ["category", "size", "occasion"]
    for k, v in entities.items():
        if isinstance(v, str):
            if k in preserve_space_keys:
                # Lowercase but keep spaces and trim extras
                new_entities[k] = v.lower().strip()  # e.g., 'Kurta Sets' -> 'kurta sets'
            else:
                # Full normalization for other keys
                new_entities[k] = v.lower().replace(" ", "").strip()
        else:
            new_entities[k] = v
    return new_entities

# --- helper: collapse variants so every product appears once ---
def dedupe_products(pinecone_data):
    grouped = {}
    for p in (pinecone_data or []):
        # Prefer a stable product id; fall back to name+tenant+rent/sale
        key = (
            p.get("product_id")
            or p.get("id")
            or ((p.get("name") or p.get("product_name") or "").strip().lower(),
                p.get("tenant_id"),
                p.get("is_rental"))
        )
        if not key:
            continue

        g = grouped.setdefault(key, {
            "base": p.copy(),
            "colors": set(),
            "sizes": set(),
            "images": set(),
            "min_price": None,
            "min_rent": None,
        })

        if p.get("color"): g["colors"].add(str(p["color"]).strip())
        if p.get("size"):  g["sizes"].add(str(p["size"]).strip())
        if p.get("image_url"): g["images"].add(p["image_url"])

        price = p.get("price")
        if isinstance(price, (int, float)):
            g["min_price"] = price if g["min_price"] is None else min(g["min_price"], price)

        rprice = p.get("rental_price")
        if isinstance(rprice, (int, float)):
            g["min_rent"] = rprice if g["min_rent"] is None else min(g["min_rent"], rprice)

    out = []
    for g in grouped.values():
        base = g["base"]
        base["available_colors"] = sorted(c for c in g["colors"] if c)
        base["available_sizes"]  = sorted(s for s in g["sizes"] if s)
        base["image_urls"] = list(g["images"])
        if g["min_price"] is not None: base["price"] = g["min_price"]
        if g["min_rent"]  is not None: base["rental_price"] = g["min_rent"]
        out.append(base)
    return out


# def normalize_entities(entities):
#     new_entities = {}
#     for k, v in entities.items():
#         if isinstance(v, str):
#             new_entities[k] = v.lower().replace(" ", "")
#         else:
#             new_entities[k] = v
#     return new_entities

async def generate_greeting_reply(language, tenant_name,session_history=None,mode: str = "call") -> str:
    # More concise, shop-aware greeting 
    emoji_instruction = "Do not use emojis." if mode == "call" else "Use emojis if you like."
    prompt = Textile_Prompt + (
        f"You are a friendly WhatsApp assistant for our {tenant_name} textile and clothing business.\n"
        f"{'Recent conversation: ' + str(session_history) if session_history else ''}\n"
        f"Greet the customer in a warm, short (1-2 sentences), conversational way in {language.upper()}. "
        f"If this is the first message, welcome them to our shop. "
        f"If ongoing, give a friendly brief follow-up. {emoji_instruction} "
        f"Only output the greeting, no product suggestions."
    )
    try:
        client = AsyncOpenAI(api_key=api_key)
        completion = await client.chat.completions.create(
            model="gpt-5-mini",
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

async def analyze_message(text: str, tenant_id: int, tenant_name: str, language: str = "en-US", intent: str | None = None, new_entities: dict | None = None, intent_confidence: float = 0.0, mode: str = "call") -> Dict[str, Any]:
    language = language
    logging.info(f"Detected language in analyze_message: {language}")
    intent_type = intent
    logging.info(f"Detected intent_type in analyze_message: {intent_type}")
    entities = new_entities or {}
    logging.info(f"Detected entities in analyze_message: {entities}")
    confidence = intent_confidence
    logging.info(f"Detected confidence in analyze_message: {confidence}")
    tenant_id = tenant_id
    logging.info(f"Tenant id ==== {tenant_id}======")
    tenant_name = tenant_name
    logging.info(f"Tenant id ==== {tenant_name}======")
    mode = mode
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
    # if intent_type in REFINEMENT_INTENTS and intent_type != "rental_inquiry" and last_main_intent:
    #     intent_type = last_main_intent
    if intent_type in MAIN_INTENTS:
        last_main_intent_by_session[tenant_id] = intent_type

    # --- Respond
    if intent_type == "greeting":
        reply = await generate_greeting_reply(language, tenant_name, session_history=history, mode=mode)
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
        # NEW: collapse variants -> one entry per product
        pinecone_data = dedupe_products(pinecone_data)
        # NEW: collect unique images (max 4)
        seen = set()
        image_urls = []
        for p in (pinecone_data or []):
            for u in (p.get("image_urls") or []):
                if u and u not in seen:
                    seen.add(u)
                    image_urls.append(u)
        image_urls = image_urls[:4]
        # --- Format product results and extra info (improved) ---
        product_lines = []
        for product in (pinecone_data or []):
            name = product.get("name") or product.get("product_name") or "Unnamed Product"

            is_rental = product.get("is_rental")
            availability = "Rent" if is_rental is True else "Sale"

            fabric = (product.get("fabric") or "").strip()
            colors = product.get("available_colors") or []
            sizes  = product.get("available_sizes") or []

            details = [availability]

            meta_bits = []
            if fabric: meta_bits.append(f"{fabric}")
            if colors: meta_bits.append("Colors: " + ", ".join(colors))
            if sizes:  meta_bits.append("Sizes: " + ", ".join(sizes))
            if meta_bits:
                details.append(" | ".join(meta_bits))

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
        reply_text = f"{products_text}\n\n{followup}"  # CHANGED: Added followup to reply_text for consistency
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
                "reply_text": reply_text,
                "media": image_urls 
            }
        elif mode == "chat":
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
                "reply_text": reply_text,
                "media": image_urls 
            }
    elif intent_type in ("availability_check"):
        # --- Extract/resolve dates ---
        start_date = None
        end_date = None

        # 1) Try entities first
        if acc_entities.get("start_date"):
            try:
                start_date = dateparser.parse(str(acc_entities["start_date"]), dayfirst=True, fuzzy=True).date()
            except Exception:
                start_date = None
        if acc_entities.get("end_date"):
            try:
                end_date = dateparser.parse(str(acc_entities["end_date"]), dayfirst=True, fuzzy=True).date()
            except Exception:
                end_date = None

        # 2) Fallback: parse from the user text like "I want rent on 15 August"
        if start_date is None:
            try:
                start_date = dateparser.parse(text, dayfirst=True, fuzzy=True).date()
            except Exception:
                reply = "❌ I couldn't understand the date. Please say a date like '15 August'."
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

        if end_date is None:
            end_date = start_date  # default to single-day rental

        # --- Figure out the variant id from collected entities ---
        variant_id = acc_entities.get("product_variant_id")

        if not variant_id:
            reply = "❌ Please select a specific product variant first."
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

        # --- Check availability in DB ---
        async with SessionLocal() as db:
            available = await is_variant_available(db, int(variant_id), start_date, end_date)

        if available:
            reply = f"✅ Available on {start_date.strftime('%d %b %Y')}."
        else:
            reply = f"❌ Not available on {start_date.strftime('%d %b %Y')}."

        # Return in the same shape as your other branches
        history.append({"role": "assistant", "content": reply})
        session_memory[tenant_id] = history
        if mode == "call":
            return {
                "input_text": text,
                "language": language,
                "intent_type": intent_type,
                "answer": reply,  # TTS uses this
                "history": history,
                "collected_entities": acc_entities
            }
        else:
            return {
                "input_text": text,
                "language": language,
                "intent_type": intent_type,
                "reply_text": reply,
                "history": history,
                "collected_entities": acc_entities
            } 
    elif intent_type == "asking_inquiry":
        
        # Get lang_root and lang_hint for language-specific response
        lang_root = (language or "en-IN").split("-")[0].lower()
        lang_hint = {
            "en": "English (India) — English only, no Hindi/Hinglish",
            "hi": "Hindi in Devanagari script — no English/Hinglish",
            "gu": "Gujarati script — no Hindi/English",
        }.get(lang_root, f"the exact locale {language}")

        # Query distinct categories from products table for the tenant
        async with SessionLocal() as db:
            result = await db.execute(
                text("SELECT DISTINCT category FROM public.products WHERE tenant_id = :tid"),
                {"tid": tenant_id}
            )
            categories = [row[0] for row in result.fetchall() if row[0]]

        if not categories:
            reply = "Sorry, we don't have any product categories available right now."
        else:
            cat_list = ", ".join(categories)
            # Use GPT to generate a natural, language-specific response
            prompt = Textile_Prompt + (
                f"You are a friendly assistant for a textile and clothing shop.\n"
                f"Generate a short, polite message listing these categories: {cat_list}.\n"
                f"Ask which category or type of product the user wants.\n"
                f"Keep it brief and natural. Do not add extra information.\n"
                f"Reply in {language.upper()}. Write in {lang_hint}. Output only the message."
            )

            client = AsyncOpenAI(api_key=api_key)
            completion = await client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are an expert, concise, friendly assistant. Respect language instructions strictly."},
                    {"role": "user", "content": prompt}
                ]
            )
            reply = completion.choices[0].message.content.strip()

        # Handle mode-specific response
        history.append({"role": "assistant", "content": reply})
        session_memory[tenant_id] = history

        if mode == "call":
            return {
                "input_text": text,
                "language": language,
                "intent_type": intent_type,
                "answer": reply,  # For TTS in call
                "history": history,
                "collected_entities": acc_entities
            }
        else:
            return {
                "input_text": text,
                "language": language,
                "intent_type": intent_type,
                "reply_text": reply,
                "history": history,
                "collected_entities": acc_entities
        }
    else:
        return Textile_Prompt
