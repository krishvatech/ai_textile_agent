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
from app.core.phase_ask_inquiry import detect_requested_attributes,fetch_distinct_options,render_attr_options,suggest_categories_advanced,scope_filters_for_attr,clean_filters_for_turn,resolve_color_list,apply_price_parsing_to_entities,fetch_starting_price,render_starting_price_single,render_starting_price_table
import json

load_dotenv()
api_key = os.getenv("GPT_API_KEY")
gpt_model = os.getenv("GPT_MODEL")
if not api_key:
    print("❌ Error: GPT_API_KEY not found in environment variables")
    exit(1)

# --- In-memory session stores (now keyed by session_key, NOT tenant_id)
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


async def generate_greeting_reply(language, tenant_name, session_history=None, mode: str = "call") -> str:
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
        # "entities_collected": {k: v for k, v in (acc_entities or {}).items() if v not in [None, "", [], {}]},
         "entities_collected": {k: v for k, v in (acc_entities or {}).items() if v not in [None, "", [], {}] and k != 'user_name'},
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
            model=gpt_model,
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


# ======================
# MAIN ORCHESTRATOR (Step-3 applied: per-customer session_key)
# ======================
async def analyze_message(
    text: str,
    tenant_id: int,
    tenant_name: str,
    language: str = "en-US",
    intent: str | None = None,
    new_entities: dict | None = None,
    intent_confidence: float = 0.0,
    mode: str = "call",
    session_key: Optional[str] = None,  # ✅ NEW
) -> Dict[str, Any]:
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

    # ---- Per-customer session key (fallback to tenant_id for backward-compat) ----
    sk = session_key or str(tenant_id)

    # Load state for this session_key
    history = session_memory.get(sk, [])
    acc_entities = session_entities.get(sk, None)
    last_main_intent = last_main_intent_by_session.get(sk, None)

    # helper to persist
    def _commit():
        session_memory[sk] = history
        session_entities[sk] = acc_entities

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

    # Persist merged entities
    session_entities[sk] = acc_entities

    print('intent_type...................', intent_type)
    # === INTENT STICKY LOGIC ===
    # if intent_type in REFINEMENT_INTENTS and intent_type != "rental_inquiry" and last_main_intent:
    #     intent_type = last_main_intent
    if intent_type in MAIN_INTENTS:
        last_main_intent_by_session[sk] = intent_type

    # --- Respond
    if intent_type == "greeting":
        reply = await generate_greeting_reply(language, tenant_name, session_history=history, mode=mode)
        history.append({"role": "assistant", "content": reply})
        _commit()
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
        if 'user_name' in filtered_entities:
            del filtered_entities['user_name']
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
            url = product.get("product_url")
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
        reply_text = f"{products_text}"

        if mode == "call":
            # Generate and speak full response
            spoken_pitch = await generate_product_pitch_prompt(language, acc_entities, pinecone_data)
            voice_response = f"{spoken_pitch} {followup}"
            history.append({"role": "assistant", "content": voice_response})
            _commit()
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
            _commit()
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
                _commit()
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
            _commit()
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
        _commit()
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
        asked_now = detect_requested_attributes(text or "")
        acc_clean = clean_filters_for_turn(acc_entities, asked_now)
        acc_clean = apply_price_parsing_to_entities(text or "", acc_clean) 
        reply = None

        # ---- Case: Explicit COLOR question with a color value (e.g., "royle blue")
        # For your #2/#5/#7/#9 flows, we return CATEGORY NAMES filtered by that color (+ other filters if present).
        if "color" in asked_now and (acc_entities or {}).get("color"):
            # Resolve possibly multi colors and/or typos, then suggest categories
            acc_local = dict(acc_entities or {})
            colors_resolved, sugg_pool = await resolve_color_list(tenant_id, acc_local, acc_local.get("color"))
            # keep as list for advanced suggester
            if colors_resolved:
                acc_local["color"] = colors_resolved  # pass list; advanced suggester accepts lists via _split_multi
            cats = await suggest_categories_advanced(tenant_id, acc_local)
            if cats:
                reply = render_categories_reply(lang_root, cats)
            else:
                if sugg_pool:
                    top = ", ".join(sugg_pool[:3])
                    reply = (f"No categories found in that color.\nDid you mean: {top}?"
                            if lang_root == "en" else
                            (f"इस रंग में कोई कैटेगरी नहीं मिली।\nक्या आपका मतलब था: {top}?"
                            if lang_root == "hi" else
                            f"આ રંગમાં કોઈ કેટેગરી મળી નથી.\nશું તમે કહ્યુ હતુ: {top}?"))
            history.append({"role": "assistant", "content": reply})
            session_memory[tenant_id] = history
            return {
                "input_text": text, "language": language, "intent_type": intent_type,
                "reply_text": reply if mode != "call" else None, "answer": reply if mode == "call" else None,
                "history": history, "collected_entities": acc_entities
            }

        # ---- Case: "what colors do you have?" (#1 variant asking for color list)
        if "color" in asked_now and not (acc_entities or {}).get("color"):
            scoped = scope_filters_for_attr(acc_entities, "color", asked_now)
            options = await fetch_distinct_options(tenant_id, scoped, "color")
            reply = render_attr_options(lang_root, "color", options) if options else \
                    ("No colors available right now." if lang_root == "en"
                    else ("अभी रंग उपलब्ध नहीं हैं." if lang_root == "hi" else "હાલમાં કોઈ કલર્સ ઉપલબ્ધ નથી."))
            history.append({"role": "assistant", "content": reply})
            session_memory[tenant_id] = history
            return {
                "input_text": text, "language": language, "intent_type": intent_type,
                "reply_text": reply if mode != "call" else None, "answer": reply if mode == "call" else None,
                "history": history, "collected_entities": acc_entities
            }

        # ---- Case: "which fabrics do you have?" (#3)
        if "fabric" in asked_now and not (acc_entities or {}).get("fabric"):
            try:
                scoped = scope_filters_for_attr(acc_entities, "fabric", asked_now)
                options = await fetch_distinct_options(tenant_id, scoped, "fabric")
                if options:
                    reply = render_attr_options(lang_root, "fabric", options)
                else:
                    reply = ("No fabrics available right now." if lang_root == "en"
                            else ("अभी फैब्रिक उपलब्ध नहीं है." if lang_root == "hi"
                                else "હાલમાં કોઈ ફેબ્રિક ઉપલબ્ધ નથી."))
            except Exception as e:
                logging.warning(f"[list-fabrics] failed: {e}")
                reply = ("No fabrics available right now." if lang_root == "en"
                        else ("अभी फैब्रिक उपलब्ध नहीं है." if lang_root == "hi"
                            else "હાલમાં કોઈ ફેબ્રિક ઉપલબ્ધ નથી."))

            history.append({"role": "assistant", "content": reply})
            session_memory[tenant_id] = history
            return {
                "input_text": text, "language": language, "intent_type": intent_type,
                "reply_text": reply if mode != "call" else None, "answer": reply if mode == "call" else None,
                "history": history, "collected_entities": acc_entities
            }
        if acc_clean.get("__starting_price__"):
            mode = acc_clean["__starting_price__"]  # 'price' or 'rental_price'
            # normalize marriage/shaadi => wedding if you do that elsewhere (optional)
            lang_root = (language or "en-IN").split("-")[0].lower()
            try:
                result = await fetch_starting_price(tenant_id, acc_clean, mode=mode)
                if isinstance(result, dict):
                    # category present
                    reply = render_starting_price_single(lang_root, result["category"], result["value"], mode)
                else:
                    # no category -> show table per category
                    reply = render_starting_price_table(lang_root, result, mode)
            except Exception as e:
                logging.warning(f"[starting-price] failed: {e}")
                reply = ("Sorry, I couldn't fetch the starting price right now."
                        if lang_root == "en" else
                        ("अभी शुरुआती कीमत नहीं मिल पाई." if lang_root == "hi" else "હવે શરૂઆતની કિંમત મેળવી શક્યો નથી."))
            history.append({"role": "assistant", "content": reply})
            session_memory[tenant_id] = history
            return {
                "input_text": text, "language": language, "intent_type": intent_type,
                "reply_text": reply if mode != "call" else None, "answer": reply if mode == "call" else None,
                "history": history, "collected_entities": acc_clean
            }
        # ---- If user provided ANY filters (fabric/color/size/price/is_rental/occasion) -> categories by ADVANCED filters
        # Any filters present? (now includes price_min/price_max or rental_price_min/rental_price_max)
        has_any_filter = any((acc_clean or {}).get(k) not in (None, "", []) for k in (
            "fabric","color","size","price","price_min","price_max",
            "rental_price","rental_price_min","rental_price_max",
            "is_rental","occasion"
        ))
        if has_any_filter:
            cats = await suggest_categories_advanced(tenant_id, acc_clean)
            if cats:
                reply = render_categories_reply(lang_root, cats)
            else:
                reply = ("No matching categories found for your filters." if lang_root == "en"
                        else ("आपके फ़िल्टर के अनुसार कोई कैटेगरी नहीं मिली." if lang_root == "hi"
                            else "તમારા ફિલ્ટર અનુસાર કોઈ કેટેગરી મળી નથી."))
            history.append({"role": "assistant", "content": reply})
            session_memory[tenant_id] = history
            return {
                "input_text": text, "language": language, "intent_type": intent_type,
                "reply_text": reply if mode != "call" else None, "answer": reply if mode == "call" else None,
                "history": history, "collected_entities": acc_clean
            }

        # ---- Default / #1: "I want to buy clothes" (no filters) -> show categories
        options = await fetch_distinct_options(tenant_id, {}, "category")
        reply = render_attr_options(lang_root, "category", options) if options else \
                ("Sorry, nothing to suggest right now."
                if lang_root == "en" else
                ("माफ़ कीजिए, अभी सुझाव उपलब्ध नहीं हैं." if lang_root == "hi"
                else "માફ કરશો, હાલમાં સૂચનો ઉપલબ્ધ નથી."))

        history.append({"role": "assistant", "content": reply})
        session_memory[tenant_id] = history
        return {
            "input_text": text, "language": language, "intent_type": intent_type,
            "reply_text": reply if mode != "call" else None, "answer": reply if mode == "call" else None,
            "history": history, "collected_entities": acc_entities
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
        _commit()

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
    else:  # NEW final fallback → behave like 'other'
        routed = await llm_route_other(text, language, tenant_id, acc_entities, history)
        reply = routed.get("reply") or (
            "How can I help you today — rental or purchase?"
            if (language or "").startswith("en") else
            ("मैं आपकी कैसे मदद कर सकता/सकती हूँ — किराये या खरीद?" if (language or "").startswith("hi")
             else "હું કેવી રીતે મદદ કરી શકું — ભાડે કે ખરીદી?")
        )
        history.append({"role": "assistant", "content": reply})
        _commit()

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
