from dotenv import load_dotenv
import os
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from dateutil import parser as dateparser
from app.core.central_system_prompt import Textile_Prompt 
import random
from sqlalchemy import text as sql_text  # Import if not already present
from app.db.session import SessionLocal
from app.core.rental_utils import is_variant_available
from app.core.product_search import pinecone_fetch_records
import json
from typing import Optional

load_dotenv()
api_key = os.getenv("GPT_API_KEY")
gpt_model = os.getenv("GPT_MODEL")
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

def _build_dynamic_heading(e: dict) -> str:
    """
    Headings built only from collected entities so far.
    Examples:
      - 'Here are saree:'
      - 'Here are rental saree:'
      - 'Here are rental saree in red:'
      - 'Here are rental saree in red with silk:'
    """
    e = {k: v for k, v in (e or {}).items() if v not in [None, "", [], {}]}

    # base parts
    parts = []
    if e.get("is_rental") is True:
        parts.append("rental")
    elif e.get("is_rental") is False:
        parts.append("sale")

    if e.get("product_url") is not None:
        parts.append(e.get("product_url"))
    elif e.get("product_url") is None:
        parts.append("")

    if e.get("category"):
        parts.append(str(e["category"]).strip())

    base = " ".join(parts) if parts else "products"

    # suffix like "in red" + "with silk" + "size M"
    suffix = []
    if e.get("color"):
        suffix.append(f"in {e['color']}")
    if e.get("fabric"):
        suffix.append(f"with {e['fabric']}")
    if e.get("size"):
        suffix.append(f"size {e['size']}")

    tail = (" " + " ".join(suffix)) if suffix else ""
    return f"Here are {base}{tail}:"

def _build_item_tags(product: dict, collected: dict) -> str:
    """
    Per-line tags using ONLY collected entities so far,
    but always include [rent]/[sale] from the product itself.
    """
    tags = []
    tags.append("rent" if product.get("is_rental") else "sale")  # always show
    for k in ("color", "fabric", "size"):
        val = collected.get(k)
        if val not in [None, "", [], {}]:
            tags.append(str(val).strip())
    return " ".join(f"- {t}" for t in tags)


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
        model=gpt_model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ],
        # gpt_model: do not send temperature/max_tokens
    )
    return completion.choices[0].message.content.strip()

def render_categories_reply(lang_root: str, categories: list[str]) -> str:
    bullets = "\n".join(f"• {c}" for c in categories[:12])  # cap to 12
    if lang_root == "hi":
        return (
            f"हमारे पास ये कैटेगरी उपलब्ध हैं:\n{bullets}\n"
            "आप किस कैटेगरी में देखना चाहेंगे? किराये या खरीद, और आपका बजट?"
        )
    if lang_root == "gu":
        return (
            f"અમારી પાસે આ કેટેગરી ઉપલબ્ધ છે:\n{bullets}\n"
            "કઈ કેટેગરીમાં જોઈએ? ભાડે કે ખરીદી, અને તમારું બજેટ?"
        )
    return (
        f"We currently carry:\n{bullets}\n"
        "Which category would you like to explore? Rental or purchase, and your budget?"
    )

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
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You are an expert, concise, friendly assistant. Respect language instructions strictly."},
            {"role": "user", "content": prompt}
        ]
        # gpt_model: don't pass temperature/max_tokens
    )
    return completion.choices[0].message.content.strip()


# NEW: language/script instruction for the LLM
def _lang_hint(language: Optional[str]) -> str:
    lr = (language or "en-IN").split("-")[0].lower()
    if lr == "hi":
        return "Reply in Hindi using Devanagari script only. No English/Hinglish."
    if lr == "gu":
        return "Reply in Gujarati script only. Keep product NAMES exactly as provided (no translation or transliteration). No Hindi/English."
    return "Reply in English (India). English only — no Hinglish."


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
            model=gpt_model,
            messages=[
                {"role": "system", "content": "You are an expert conversation starter and friendly textile assistant."},
                {"role": "user", "content": prompt}
            ]
            # gpt_model: don't pass temperature/max_tokens
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

# NEW: LLM-driven router for 'other'
async def llm_route_other(
    text: str,
    language: Optional[str],
    tenant_id: int,
    acc_entities: Dict[str, Any],
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Uses gpt_model (JSON mode) to:
      - understand user's message + recent history + collected entities
      - (optionally) show a short help menu using live categories
      - produce a final reply (≤ ~80 words) with at most ONE follow-up question
    Returns:
      { action: 'smalltalk'|'help'|'followup'|'handoff'|'unknown',
        reply: str,
        ask_fields: list[str] (≤3),
        confidence: float }
    """
    # 1) Pull live categories (helps the model craft a helpful 'help' reply)
    categories: List[str] = []
    try:
        async with SessionLocal() as db:
            result = await db.execute(
                sql_text("SELECT DISTINCT category FROM public.products WHERE tenant_id = :tid"),
                {"tid": tenant_id}
            )
            categories = [row[0] for row in result.fetchall() if row[0]]
    except Exception as e:
        logging.warning(f"Category fetch failed: {e}")

    recent_history = (history or [])[-6:]

    sys_msg = (
        "You are a warm, concise shop assistant for an Indian textile store (retail + wholesale + rentals).\n"
        "RULES:\n"
        "• Follow the locale instruction exactly (script + no transliteration).\n"
        "• Ask at most ONE follow-up question.\n"
        "• Never invent stock, sizes, fabrics, colors, prices, dates, or offers.\n"
        "• If you mention product NAMES, keep them EXACTLY as provided (no translation/transliteration).\n"
        "• Keep replies short and natural for WhatsApp/voice (≤ ~80 words).\n"
        "• If user asks for a human or seems upset, choose action='handoff' and write a polite line.\n"
        "• If the user asks “What’s my name?” and known_profile.name exists, say it exactly; else say you don’t have it and ask once to share it.\n"
        "\nFINAL OUTPUT FORMAT:\n"
        "• Return ONLY a JSON object (json) with keys: action, reply, ask_fields, confidence.\n"
        "• No preamble, no code fences, no extra text — just valid JSON.\n"
    )

    user_payload = {
        "locale_instruction": _lang_hint(language),
        "business_prompt": Textile_Prompt,  # your global policy block
        "user_message": text,
        "entities_collected": {k: v for k, v in (acc_entities or {}).items() if v not in [None, "", [], {}]},
        "recent_history": recent_history,
        "categories": categories[:12],
        "known_profile": {
            "name": (acc_entities or {}).get("user_name") or ""
        },
        "allowed_actions": ["smalltalk","help","followup","handoff","unknown"],
        "allowed_followup_fields": [
            "is_rental","occasion","fabric","size","color","category",
            "product_name","quantity","location","type","price","rental_price",
            "user_name"
        ],
        "output_contract": {
            "action": "one of allowed_actions",
            "reply": "final user-facing text in the correct language, one question max",
            "ask_fields": "<=3 field names from allowed_followup_fields (or empty)",
            "confidence": "float 0..1"
        }
    }

    try:
        client = AsyncOpenAI(api_key=api_key)
        completion = await client.chat.completions.create(
            model=gpt_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                # Safety nudge so 'json' definitely appears in messages:
                {"role": "system", "content": "Remember: respond with JSON only — a single JSON object."}
            ],
        )

        data = json.loads(completion.choices[0].message.content)
    except Exception as e:
        logging.warning(f"LLM router JSON parse/error: {e}")
        # Safe language-aware help fallback
        lr = (language or "en-IN").split("-")[0].lower()
        fallback = render_categories_reply(lr, categories or ["Sarees","Kurta Sets","Lehengas","Blouses","Gowns"])
        return {"action": "help", "reply": fallback, "ask_fields": [], "confidence": 0.4}

    # Defaults + clamps
    data.setdefault("action", "unknown")
    data.setdefault("reply", "")
    data.setdefault("ask_fields", [])
    data.setdefault("confidence", 0.6)
    if isinstance(data.get("ask_fields"), list) and len(data["ask_fields"]) > 3:
        data["ask_fields"] = data["ask_fields"][:3]

    return data

# NEW: extract a user's name from free text (LLM, JSON mode)
async def extract_user_name(text: str, language: Optional[str]) -> Optional[str]:
    """
    Ask gpt-5-mini (JSON mode) to see if the message states the user's name.
    Returns a clean display name or None.
    """
    # Quick short-circuit: empty or trivial text → skip
    if not text or len(text.strip()) < 2:
        return None

    sys_msg = (
        "You are a precise NER helper.\n"
        "Identify if the user is stating THEIR OWN NAME in the message.\n"
        "Only return JSON (json) with keys: has_name (bool), name (string).\n"
        "The 'name' must be the person's display name as said (e.g., 'Avinash' or 'Avinash Od').\n"
        "If unsure, has_name=false and name=\"\".\n"
    )
    user_payload = {
        "message": text,
        "locale": language,
        "examples": [
            {"text":"I'm Avinash","has_name":True,"name":"Avinash"},
            {"text":"My name is Avinash Od","has_name":True,"name":"Avinash Od"},
            {"text":"This is Priya here","has_name":True,"name":"Priya"},
            {"text":"What's my name?","has_name":False,"name":""},
            {"text":"Ok thanks","has_name":False,"name":""}
        ],
        "output_contract": {"has_name":"bool","name":"string"}
    }
    try:
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model="gpt-5-mini",
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content":sys_msg},
                {"role":"user","content":json.dumps(user_payload, ensure_ascii=False)},
                {"role":"system","content":"Respond with JSON only — a single JSON object."}  # ensures 'json' appears
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        if data.get("has_name") and isinstance(data.get("name"), str) and data["name"].strip():
            # Basic cleanup: collapse inner spaces
            name = " ".join(data["name"].strip().split())
            # Cap length to avoid prompt injection
            if len(name) <= 60:
                return name
    except Exception as e:
        logging.warning(f"extract_user_name failed: {e}")
    return None

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
    
    if not acc_entities.get("user_name"):
        maybe_name = await extract_user_name(text, language)
        if maybe_name:
            acc_entities["user_name"] = maybe_name

    # --- Rental/Purchase logic
    is_rental = acc_entities.get("is_rental")

    if is_rental is True:
        acc_entities.pop("price", None)        # Remove 'price' if present
    elif is_rental is False:
        acc_entities.pop("rental_price", None) # Remove 'rental_price' if present

    session_entities[tenant_id] = acc_entities

    print('intent_type...................', intent_type)
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
        # 1) Collect and normalize entities for Pinecone filtering
        filtered_entities = filter_non_empty_entities(acc_entities)
        filtered_entities_norm = normalize_entities(filtered_entities)
        filtered_entities_norm = clean_entities_for_pinecone(filtered_entities_norm)

        # 2) Fetch + dedupe products
        pinecone_data = await pinecone_fetch_records(filtered_entities_norm, tenant_id)
        pinecone_data = dedupe_products(pinecone_data)

        # 3) (Optional) collect a few unique images
        seen, image_urls = set(), []
        for p in (pinecone_data or []):
            for u in (p.get("image_urls") or []):
                if u and u not in seen:
                    seen.add(u)
                    image_urls.append(u)
        image_urls = image_urls[:4]

        # 4) Build heading + lines using ONLY collected entities so far
        collected_for_text = {
            k: v for k, v in (filtered_entities or {}).items()
            if k in ("category", "color", "fabric", "size", "is_rental") and v not in (None, "", [], {})
        }

        heading = _build_dynamic_heading(collected_for_text)

        product_lines = []
        for product in (pinecone_data or []):
            # Name
            name = product.get("name") or product.get("product_name") or "Unnamed Product"

            # Tags: always include [rent]/[sale], plus collected entities only
            tags = _build_item_tags(product, collected_for_text)

            # URL (with simple normalization + fallbacks)
            url = (
                product.get("product_url")
            )
            if isinstance(url, str):
                url = url.strip()
                if url and not url.startswith(("http://", "https://")):
                    url = "https://" + url.lstrip("/")

            # Final line (include URL only if present)
            product_lines.append(f"- {name} {tags}" + (f" — {url}" if url else ""))

        # 5) Final message
        products_text = (
            f"{heading}\n" + "\n".join(product_lines)
            if product_lines else
            "Sorry, no products match your search so far."
        )

        followup = await FollowUP_Question(intent_type, acc_entities, language, session_history=history)
        reply_text = f"{products_text}\n\n{followup}"


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
    elif intent_type == "availability_check":
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
        lang_root = (language or "en-IN").split("-")[0].lower()
        lang_hint = {
            "en": "English (India) — English only, no Hinglish.",
            "hi": "Hindi (Devanagari) only — no English/Hinglish.",
            "gu": "Gujarati script only — no Hindi/English.",
        }.get(lang_root, f"Use the exact locale: {language}")

        async with SessionLocal() as db:
            result = await db.execute(
                sql_text("SELECT DISTINCT category FROM public.products WHERE tenant_id = :tid"),
                {"tid": tenant_id}
            )
            categories = [row[0] for row in result.fetchall() if row[0]]

        if not categories:
            reply = (
                "Sorry, we don't have any product categories available right now."
                if lang_root == "en" else
                ("माफ़ कीजिए, अभी कोई कैटेगरी उपलब्ध नहीं है." if lang_root == "hi"
                else "માફ કરશો, હાલમાં કોઈ કેટેગરી ઉપલબ્ધ નથી.")
            )
        else:
            # 1) Deterministic fallback (always works)
            reply = render_categories_reply(lang_root, categories)

            # 2) Try GPT to polish — keep only if it actually mentions a category
            try:
                cat_list = ", ".join(sorted(set(categories)))
                prompt = Textile_Prompt
                client = AsyncOpenAI(api_key=api_key)
                completion = await client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                gpt_reply = completion.choices[0].message.content.strip()

                # Use GPT reply only if it includes at least one known category (prevents generic greetings)
                if any(c.lower() in gpt_reply.lower() for c in categories[:3]):
                    reply = gpt_reply
            except Exception as e:
                logging.warning(f"GPT polish failed, using fallback. Error: {e}")

        history.append({"role": "assistant", "content": reply})
        session_memory[tenant_id] = history

        if mode == "call":
            return {
                "input_text": text,          # rename from `text` to avoid shadowing
                "language": language,
                "intent_type": intent_type,
                "answer": reply,
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
    elif intent_type == "fabric_inquiry":
        # --- language setup ---
        lang_root = (language or "en-IN").split("-")[0].lower()

        def render_fabrics_reply(lang: str, fabrics: list[str]) -> str:
            csv = ", ".join(fabrics)
            if lang == "hi":
                return f"उपलब्ध फैब्रिक्स: {csv}. आप किसे पसंद करेंगे?"
            if lang == "gu":
                return f"ઉપલબ્ધ ફેબ્રિક: {csv}. તમે કયું પસંદ કરશો?"
            return f"Available fabrics: {csv}. Which one do you prefer?"

        # --- optional filters from your parsed entities ---
        category     = (acc_entities or {}).get("category")
        product_type = (acc_entities or {}).get("type")
        is_rental    = (acc_entities or {}).get("is_rental")

        where = [
            "p.tenant_id = :tid",
            "COALESCE(pv.is_active, TRUE) = TRUE",
        ]
        params = {"tid": tenant_id}

        if category:
            where.append("LOWER(p.category) = LOWER(:category)")
            params["category"] = str(category)
        if product_type:
            where.append("LOWER(p.type) = LOWER(:ptype)")
            params["ptype"] = str(product_type)
        if is_rental is not None:
            where.append("pv.is_rental = :is_rental")
            params["is_rental"] = bool(is_rental)

        sql = f"""
            SELECT DISTINCT pv.fabric
            FROM public.product_variants pv
            JOIN public.products p ON p.id = pv.product_id
            WHERE {' AND '.join(where)}
            ORDER BY pv.fabric
        """

        # --- fetch fabrics ---
        async with SessionLocal() as db:
            result = await db.execute(sql_text(sql), params)
            fabrics_raw = [row[0] for row in result.fetchall() if row[0]]

        # normalize & dedupe (trim, title-case)
        fabrics = sorted({str(f).strip().title() for f in fabrics_raw if str(f).strip()})

        if not fabrics:
            reply = (
                "Sorry, no fabrics are available right now."
                if lang_root == "en" else
                ("माफ़ कीजिए, अभी कोई फ़ैब्रिक उपलब्ध नहीं है." if lang_root == "hi"
                else "માફ કરશો, હાલમાં કોઈ ફેબ્રિક ઉપલબ્ધ નથી.")
            )
        else:
            reply = render_fabrics_reply(lang_root, fabrics)
            try:
                client = AsyncOpenAI(api_key=api_key)
                prompt = (
                    f"Language: {language}. "
                    f"Fabrics available: {', '.join(fabrics)}. "
                    "Write one short friendly line listing them and asking the user to choose. "
                    "Do not add anything else."
                )
                completion = await client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                gpt_reply = (completion.choices[0].message.content or "").strip()
                if any(f.lower() in gpt_reply.lower() for f in fabrics[:3]):
                    reply = gpt_reply
            except Exception as e:
                logging.warning(f"GPT polish failed; using fallback. Error: {e}")

        # --- record + return ---
        history.append({"role": "assistant", "content": reply})
        session_memory[tenant_id] = history

        if mode == "call":
            return {
                "language": language,
                "intent_type": intent_type,
                "answer": reply,
                "history": history,
                "collected_entities": acc_entities
            }
        else:
            return {
                "language": language,
                "intent_type": intent_type,
                "reply_text": reply,
                "history": history,
                "collected_entities": acc_entities
            }        
    elif intent_type == "other" or intent_type is None:  # NEW
        routed = await llm_route_other(text, language, tenant_id, acc_entities, history)
        # Simple language-aware fallback if router returned empty reply
        if not routed.get("reply"):
            if (language or "").lower().startswith("hi"):
                reply = "मैं आपकी कैसे मदद कर सकता/सकती हूँ — किराये या खरीद?"
            elif (language or "").lower().startswith("gu"):
                reply = "હું કેવી રીતે મદદ કરી શકું — ભાડે કે ખરીદી?"
            else:
                reply = "How can I help you today — rental or purchase?"
        else:
            reply = routed["reply"]

        history.append({"role": "assistant", "content": reply})
        session_memory[tenant_id] = history

        if mode == "call":
            return {
                "input_text": text,
                "language": language,
                "intent_type": "other",
                "answer": reply,   # TTS payload
                "history": history,
                "collected_entities": acc_entities,
                "router": routed
            }
        else:
            return {
                "input_text": text,
                "language": language,
                "intent_type": "other",
                "reply_text": reply,
                "history": history,
                "collected_entities": acc_entities,
                "router": routed
            }
    elif intent_type == "price_inquiry":
        # --- language setup ---
        lang_root = (language or "en-IN").split("-")[0].lower()

        def render_price_reply(lang: str, min_price: float, max_price: float, category: str) -> str:
            if lang == "hi":
                return f"{category} श्रेणी में उपलब्ध उत्पादों की कीमत {min_price} से {max_price} तक है। क्या आप और विवरण चाहते हैं?"
            if lang == "gu":
                return f"{category} શ્રેણીમાં ઉપલબ્ધ ઉત્પાદનોની કિંમત {min_price} થી {max_price} સુધી છે. શું તમે વધુ વિગતો માંગો છો?"
            return f"Prices for {category} category range from {min_price} to {max_price}. Would you like more details?"

        # --- optional filters from your parsed entities ---
        category = (acc_entities or {}).get("category")
        product_type = (acc_entities or {}).get("type")
        is_rental = (acc_entities or {}).get("is_rental")

        if not category:
            # Fallback if no category is provided
            reply = (
                "Please specify a category to check prices."
                if lang_root == "en" else
                ("कृपया कीमत जांचने के लिए एक श्रेणी निर्दिष्ट करें." if lang_root == "hi"
                else "કૃપા કરીને કિંમત તપાસવા માટે એક શ્રેણી નિર્દિષ્ટ કરો.")
            )
            history.append({"role": "assistant", "content": reply})
            session_memory[tenant_id] = history

            if mode == "call":
                return {
                    "language": language,
                    "intent_type": intent_type,
                    "answer": reply,
                    "history": history,
                    "collected_entities": acc_entities
                }
            else:
                return {
                    "language": language,
                    "intent_type": intent_type,
                    "reply_text": reply,
                    "history": history,
                    "collected_entities": acc_entities
                }

        where = [
            "p.tenant_id = :tid",
            "COALESCE(pv.is_active, TRUE) = TRUE",
            "LOWER(p.category) = LOWER(:category)"
        ]
        params = {"tid": tenant_id, "category": str(category)}

        if product_type:
            where.append("LOWER(p.type) = LOWER(:ptype)")
            params["ptype"] = str(product_type)
        if is_rental is not None:
            where.append("pv.is_rental = :is_rental")
            params["is_rental"] = bool(is_rental)

        price_field = "pv.rental_price" if is_rental else "pv.price"

        sql = f"""
            SELECT MIN({price_field}) AS min_price, MAX({price_field}) AS max_price
            FROM public.product_variants pv
            JOIN public.products p ON p.id = pv.product_id
            WHERE {' AND '.join(where)}
        """

        # --- fetch prices ---
        async with SessionLocal() as db:
            result = await db.execute(sql_text(sql), params)
            row = result.fetchone()
            min_price = row[0] if row else None
            max_price = row[1] if row else None

        if min_price is None or max_price is None:
            reply = (
                f"Sorry, no prices available for {category} right now."
                if lang_root == "en" else
                (f"माफ़ कीजिए, {category} के लिए अभी कोई कीमत उपलब्ध नहीं है." if lang_root == "hi"
                else f"માફ કરશો, {category} માટે હાલમાં કોઈ કિંમત ઉપલબ્ધ નથી.")
            )
        else:
            reply = render_price_reply(lang_root, min_price, max_price, category)
            try:
                prompt = Textile_Prompt+ (
                    f"Language: {language}. "
                    f"Category: {category}. Min price: {min_price}. Max price: {max_price}. "
                    "Write one short friendly line stating the price range and asking if they want more details. "
                    "Do not add anything else."
                )
                completion = await client.chat.completions.create(
                    model=gpt_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                gpt_reply = (completion.choices[0].message.content or "").strip()
                if str(min_price).lower() in gpt_reply.lower() or str(max_price).lower() in gpt_reply.lower():
                    reply = gpt_reply
            except Exception as e:
                logging.warning(f"GPT polish failed; using fallback. Error: {e}")

        # --- record + return ---
        history.append({"role": "assistant", "content": reply})
        session_memory[tenant_id] = history

        if mode == "call":
            return {
                "language": language,
                "intent_type": intent_type,
                "answer": reply,
                "history": history,
                "collected_entities": acc_entities
            }
        else:
            return {
                "language": language,
                "intent_type": intent_type,
                "reply_text": reply,
                "history": history,
                "collected_entities": acc_entities
            }
    elif intent_type == "size_query":
        # --- language setup ---
        lang_root = (language or "en-IN").split("-")[0].lower()

        def render_sizes_reply(lang: str, sizes: list[str], category: str) -> str:
            csv = ", ".join(sizes)
            if lang == "hi":
                return f"{category} श्रेणी में उपलब्ध आकार: {csv}. आप किसे पसंद करेंगे?"
            if lang == "gu":
                return f"{category} શ્રેણીમાં ઉપલબ્ધ કદ: {csv}. તમે કયું પસંદ કરશો?"
            return f"Available sizes for {category}: {csv}. Which one do you prefer?"

        # --- optional filters from your parsed entities ---
        category = (acc_entities or {}).get("category")
        product_type = (acc_entities or {}).get("type")
        is_rental = (acc_entities or {}).get("is_rental")

        if not category:
            # Fallback if no category is provided
            reply = (
                "Please specify a category to check sizes."
                if lang_root == "en" else
                ("कृपया आकार जांचने के लिए एक श्रेणी निर्दिष्ट करें." if lang_root == "hi"
                else "કૃપા કરીને કદ તપાસવા માટે એક શ્રેણી નિર્દિષ્ટ કરો.")
            )
            history.append({"role": "assistant", "content": reply})
            session_memory[tenant_id] = history

            if mode == "call":
                return {
                    "language": language,
                    "intent_type": intent_type,
                    "answer": reply,
                    "history": history,
                    "collected_entities": acc_entities
                }
            else:
                return {
                    "language": language,
                    "intent_type": intent_type,
                    "reply_text": reply,
                    "history": history,
                    "collected_entities": acc_entities
                }

        where = [
            "p.tenant_id = :tid",
            "COALESCE(pv.is_active, TRUE) = TRUE",
            "LOWER(p.category) = LOWER(:category)"
        ]
        params = {"tid": tenant_id, "category": str(category)}

        if product_type:
            where.append("LOWER(p.type) = LOWER(:ptype)")
            params["ptype"] = str(product_type)
        if is_rental is not None:
            where.append("pv.is_rental = :is_rental")
            params["is_rental"] = bool(is_rental)

        sql = f"""
            SELECT DISTINCT pv.size
            FROM public.product_variants pv
            JOIN public.products p ON p.id = pv.product_id
            WHERE {' AND '.join(where)}
            ORDER BY pv.size
        """

        # --- fetch sizes ---
        async with SessionLocal() as db:
            result = await db.execute(sql_text(sql), params)
            sizes_raw = [row[0] for row in result.fetchall() if row[0]]

        # normalize & dedupe (trim, title-case)
        sizes = sorted({str(s).strip().title() for s in sizes_raw if str(s).strip()})

        if not sizes:
            reply = (
                f"Sorry, no sizes available for {category} right now."
                if lang_root == "en" else
                (f"माफ़ कीजिए, {category} के लिए अभी कोई आकार उपलब्ध नहीं है." if lang_root == "hi"
                else f"માફ કરશો, {category} માટે હાલમાં કોઈ કદ ઉપલબ્ધ નથી.")
            )
        else:
            reply = render_sizes_reply(lang_root, sizes, category)
            try:
                prompt = Textile_Prompt + (
                    f"Language: {language}. "
                    f"Category: {category}. Sizes available: {', '.join(sizes)}. "
                    "Write one short friendly line listing them and asking the user to choose. "
                    "Do not add anything else."
                )
                completion = await client.chat.completions.create(
                    model=gpt_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                gpt_reply = (completion.choices[0].message.content or "").strip()
                if any(s.lower() in gpt_reply.lower() for s in sizes[:3]):
                    reply = gpt_reply
            except Exception as e:
                logging.warning(f"GPT polish failed; using fallback. Error: {e}")

        # --- record + return ---
        history.append({"role": "assistant", "content": reply})
        session_memory[tenant_id] = history

        if mode == "call":
            return {
                "language": language,
                "intent_type": intent_type,
                "answer": reply,
                "history": history,
                "collected_entities": acc_entities
            }
        else:
            return {
                "language": language,
                "intent_type": intent_type,
                "reply_text": reply,
                "history": history,
                "collected_entities": acc_entities
            }
    elif intent_type == "color_preference":
        # --- language setup ---
        lang_root = (language or "en-IN").split("-")[0].lower()

        def render_colors_reply(lang: str, colors: list[str], category: str) -> str:
            csv = ", ".join(colors)
            if lang == "hi":
                return f"{category} श्रेणी में उपलब्ध रंग: {csv}. आप किसे पसंद करेंगे?"
            if lang == "gu":
                return f"{category} શ્રેણીમાં ઉપલબ્ધ રંગો: {csv}. તમે કયું પસંદ કરશો?"
            return f"Available colors for {category}: {csv}. Which one do you prefer?"

        # --- optional filters from your parsed entities ---
        category = (acc_entities or {}).get("category")
        product_type = (acc_entities or {}).get("type")
        is_rental = (acc_entities or {}).get("is_rental")

        if not category:
            # Fallback if no category is provided
            reply = (
                "Please specify a category to check colors."
                if lang_root == "en" else
                ("कृपया रंग जांचने के लिए एक श्रेणी निर्दिष्ट करें." if lang_root == "hi"
                else "કૃપા કરીને રંગ તપાસવા માટે એક શ્રેણી નિર્દિષ્ટ કરો.")
            )
            history.append({"role": "assistant", "content": reply})
            session_memory[tenant_id] = history

            if mode == "call":
                return {
                    "language": language,
                    "intent_type": intent_type,
                    "answer": reply,
                    "history": history,
                    "collected_entities": acc_entities
                }
            else:
                return {
                    "language": language,
                    "intent_type": intent_type,
                    "reply_text": reply,
                    "history": history,
                    "collected_entities": acc_entities
                }

        where = [
            "p.tenant_id = :tid",
            "COALESCE(pv.is_active, TRUE) = TRUE",
            "LOWER(p.category) = LOWER(:category)"
        ]
        params = {"tid": tenant_id, "category": str(category)}

        if product_type:
            where.append("LOWER(p.type) = LOWER(:ptype)")
            params["ptype"] = str(product_type)
        if is_rental is not None:
            where.append("pv.is_rental = :is_rental")
            params["is_rental"] = bool(is_rental)

        sql = f"""
            SELECT DISTINCT pv.color
            FROM public.product_variants pv
            JOIN public.products p ON p.id = pv.product_id
            WHERE {' AND '.join(where)}
            ORDER BY pv.color
        """

        # --- fetch colors ---
        async with SessionLocal() as db:
            result = await db.execute(sql_text(sql), params)
            colors_raw = [row[0] for row in result.fetchall() if row[0]]

        # normalize & dedupe (trim, title-case)
        colors = sorted({str(c).strip().title() for c in colors_raw if str(c).strip()})

        if not colors:
            reply = (
                f"Sorry, no colors available for {category} right now."
                if lang_root == "en" else
                (f"माफ़ कीजिए, {category} के लिए अभी कोई रंग उपलब्ध नहीं है." if lang_root == "hi"
                else f"માફ કરશો, {category} માટે હાલમાં કોઈ રંગ ઉપલબ્ધ નથી.")
            )
        else:
            reply = render_colors_reply(lang_root, colors, category)
            try:
                prompt = Textile_Prompt + (
                    f"Language: {language}. "
                    f"Category: {category}. Colors available: {', '.join(colors)}. "
                    "Write one short friendly line listing them and asking the user to choose. "
                    "Do not add anything else."
                )
                completion = await client.chat.completions.create(
                    model=gpt_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                gpt_reply = (completion.choices[0].message.content or "").strip()
                if any(c.lower() in gpt_reply.lower() for c in colors[:3]):
                    reply = gpt_reply
            except Exception as e:
                logging.warning(f"GPT polish failed; using fallback. Error: {e}")

        # --- record + return ---
        history.append({"role": "assistant", "content": reply})
        session_memory[tenant_id] = history

        if mode == "call":
            return {
                "language": language,
                "intent_type": intent_type,
                "answer": reply,
                "history": history,
                "collected_entities": acc_entities
            }
        else:
            return {
                "language": language,
                "intent_type": intent_type,
                "reply_text": reply,
                "history": history,
                "collected_entities": acc_entities
            }

    else:  # NEW final fallback → behave like 'other'
        routed = await llm_route_other(text, language, tenant_id, acc_entities, history)
        reply = routed.get("reply") or (
            "How can I help you today — rental or purchase?"
            if (language or "").startswith("en") else
            ("मैं आपकी कैसे मदद कर सकता/सकती हूँ — किराये या खरीद?" if (language or "").startswith("hi")
             else "હું કેવી રીતે મદદ કરી શકું — ભાડે કે ખરીદી?")
        )
        history.append({"role": "assistant", "content": reply})
        session_memory[tenant_id] = history

        if mode == "call":
            return {
                "input_text": text,
                "language": language,
                "intent_type": intent_type or "other",
                "answer": reply,
                "history": history,
                "collected_entities": acc_entities,
                "router": routed
            }
        else:
            return {
                "input_text": text,
                "language": language,
                "intent_type": intent_type or "other",
                "reply_text": reply,
                "history": history,
                "collected_entities": acc_entities,
                "router": routed
            }

