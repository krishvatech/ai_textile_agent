from dotenv import load_dotenv
import os
import re
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
from app.core.phase_ask_inquiry import detect_requested_attributes,fetch_distinct_options,apply_price_parsing_to_entities,fetch_starting_price,render_starting_price_single,render_starting_price_table,_price_mode_from_text_and_entities,render_options_reply,_build_product_lines,clean_filters_for_turn
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
    "asking_inquiry", "color_preference", "size_query", "fabric_inquiry",
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
      - 'Here are saree:'
      - 'Here are rental saree:'
      - 'Here are rental saree for wedding in red with silk size M:'
    """
    e = {k: v for k, v in (e or {}).items() if v not in [None, "", [], {}]}

    # base parts
    parts = []
    if e.get("is_rental") is True:
        parts.append("rental")
    elif e.get("is_rental") is False:
        parts.append("sale")

    if e.get("category"):
        parts.append(str(e["category"]).strip())

    base = " ".join(parts) if parts else "products"

    # suffix chips: occasion, color, fabric, size
    suffix = []

    # occasion can be str or list
    occ = e.get("occasion")
    if isinstance(occ, list) and occ:
        occ = occ[0]
    if occ:
        suffix.append(f"for {occ}")

    if e.get("color"):
        suffix.append(f"in {e['color']}")
    if e.get("fabric"):
        suffix.append(f"with {e['fabric']}")

    # ✅ show size only if not "Freesize" (handles Free size / One Size, etc.)
    size_val = str(e.get("size") or "").strip()
    if size_val and size_val.lower() != "Freesize":
        suffix.append(f"size {size_val}")

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
    def _is_missing(val):
        return (val is None) or (val == "") or (isinstance(val, (list, dict)) and not val)

    is_rental_val = entities.get("is_rental", None)
    base_keys = [
        "is_rental", "occasion", "fabric", "size", "color", "category",
        "product_name", "quantity", "location", "type"
    ]
    # Only ask for the correct price field
    price_keys = ["rental_price"] if is_rental_val is True else (["price"] if is_rental_val is False else ["price", "rental_price"])

    # This is the canonical ordered set we will evaluate for missing-ness
    entity_priority = base_keys + price_keys

    # Find missing fields (consider empty as missing)
    missing_fields = [k for k in entity_priority if _is_missing(entities.get(k))]

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

def _infer_is_rental_from_text(text: str, acc_entities: dict) -> None:
    """
    If is_rental not set, infer from keywords in the current user text.
    Idempotent; mutates acc_entities in place.
    """
    if acc_entities.get("is_rental") in (None, "", [], {}):
        s = (text or "").lower()
        # rent-side triggers (EN + hi + gu + common variants)
        if re.search(r"\b(rent|rental|on\s*rent|for\s*rent|hire|kiraye|kiraye\s*pe|भाड़े|भाड़े|भाड़े\s*पर|ભાડે)\b", s):
            acc_entities["is_rental"] = True
        # buy-side triggers
        elif re.search(r"\b(buy|purchase|kharee?d|खरीद(?:ना)?)\b", s):
            acc_entities["is_rental"] = False

def _infer_occasion_from_text(text: str, acc_entities: dict) -> None:
    """
    If occasion not set, map common phrases to: wedding|party|festival|casual.
    Mutates acc_entities in place.
    """
    if (acc_entities or {}).get("occasion") not in (None, "", [], {}):
        return

    s = (text or "").lower()

    # Hindi/Gujarati + English triggers (keep this small and precise)
    if any(x in s for x in ["wedding", "shaadi", "shadi", "vivah", "baraat", "reception", "pher", "pherā", "pheras"]):
        acc_entities["occasion"] = "wedding"
        return
    if any(x in s for x in ["party", "partywear", "party wear"]):
        acc_entities["occasion"] = "party"
        return
    if any(x in s for x in ["festival", "festive", "diwali", "navratri", "eid"]):
        acc_entities["occasion"] = "festival"
        return
    if any(x in s for x in ["daily wear", "casual", "office", "regular"]):
        acc_entities["occasion"] = "casual"
        return

# --- Heuristic backfill for a facet (e.g., fabric) from user's short message ---
async def _backfill_facet_from_text(text: str, tenant_id: int, ctx: dict, facet: str) -> str | None:
    """
    If the user sends a very short message like 'georgette' or 'silk',
    try to match it to a known option for the given facet (fabric/color/size).
    Returns the canonical option string if matched, else None.
    """
    raw = (text or "").strip()
    if not raw or len(raw.split()) > 3:  # only try for short inputs
        return None

    # fetch available options for this facet in current context (category/rent/occasion, etc.)
    options = await fetch_distinct_options(tenant_id, ctx or {}, facet, limit=60)

    def _norm(s: str) -> str:
        return re.sub(r"[\s\._-]+", " ", (s or "").strip().lower())

    needle = _norm(raw)
    for opt in options or []:
        if _norm(opt) == needle or needle in _norm(opt):
            return opt
    return None


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
    detected_intent = intent_type
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
    prev_category = (acc_entities or {}).get("category")

    # --- Hard reset on category switch: keep ONLY the new category
    new_category_from_turn = (entities or {}).get("category")
    if new_category_from_turn:
        if (prev_category is None) or (
            str(new_category_from_turn).strip().lower()
            != str(prev_category or "").strip().lower()
        ):
            # reset to the new category, but keep name and (optionally) rent/buy
            acc_entities = {
                "category": new_category_from_turn,
                # uncomment next line if you want rent/buy to persist across category switches
                # "is_rental": (acc_entities or {}).get("is_rental"),
            }

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
    
    # 🔎 Infer rent/buy from the current turn before any branching/cleanups
    _infer_is_rental_from_text(text, acc_entities)

    # 🔎 Infer occasion the same way (e.g., "for wedding")
    _infer_occasion_from_text(text, acc_entities)

    # 🔎 Backfill FABRIC if user sent a short message like "georgette"
    if not acc_entities.get("fabric"):
        guessed_fabric = await _backfill_facet_from_text(text, tenant_id, acc_entities, "fabric")
        if guessed_fabric:
            acc_entities["fabric"] = guessed_fabric

    # --- Rental/Purchase logic
    is_rental = acc_entities.get("is_rental")

    if is_rental is True:
        acc_entities.pop("price", None)        # Remove 'price' if present
    elif is_rental is False:
        acc_entities.pop("rental_price", None) # Remove 'rental_price' if present

    # Persist merged entities
    sp_probe = apply_price_parsing_to_entities(text or "", acc_entities)
    if sp_probe.get("__starting_price__"):
        # merge back everything EXCEPT the private flag
        acc_entities.update({k: v for k, v in sp_probe.items() if k != "__starting_price__"})

        # For true "starting", drop granular filters so min isn't inflated
        local_sp = dict(acc_entities)
        for k in ("size", "color", "fabric", "occasion"):
            local_sp.pop(k, None)

        lang_root = (language or "en-IN").split("-")[0].lower()
        price_mode = _price_mode_from_text_and_entities(text, local_sp)  # 'price' or 'rental_price'
        sp = await fetch_starting_price(tenant_id, local_sp, price_mode)

        if isinstance(sp, dict):  # category present
            reply = render_starting_price_single(
                lang_root,
                sp.get("category") or "Items",
                sp.get("value"),
                price_mode
            )
        else:  # no category -> table by category
            reply = render_starting_price_table(lang_root, sp or [], price_mode)
        
        acc_entities.pop("__starting_price__", None)
        local_sp.pop("__starting_price__", None)

        history.append({"role": "assistant", "content": reply})
        _commit()
        return {
            "input_text": text,
            "language": language,
            "intent_type": intent_type or "asking_inquiry",
            "reply_text": reply if mode != "call" else None,
            "answer": reply if mode == "call" else None,
            "history": history,
            "collected_entities": local_sp
        }

    print('intent_type...................', intent_type)

    # ---- What did the user ask in this turn? (needed by sticky logic & guards)
    asked_now = detect_requested_attributes(text or "")
    if (new_entities or {}).get("category") and "category" not in asked_now:
        asked_now.append("category")
    generic_category_ask = ("category" in asked_now) and not ((new_entities or {}).get("category"))

    # === INTENT STICKY LOGIC ===
    # if intent_type in REFINEMENT_INTENTS and intent_type != "rental_inquiry" and last_main_intent:
    #     intent_type = last_main_intent

    # ---- Contextual upgrade: attribute-only messages become product_search when a category is known or we were already searching
    if intent_type in (None, "other"):
        has_attr = any(
            (acc_entities.get(k) not in (None, "", [], {}))
            for k in ("occasion", "fabric", "color", "size", "is_rental", "price", "rental_price")
        )
        if has_attr and ((acc_entities.get("category")) or (last_main_intent == "product_search")):
            intent_type = "product_search"

    if (intent_type in REFINEMENT_INTENTS) and (last_main_intent in MAIN_INTENTS) and not generic_category_ask:
        intent_type = last_main_intent
    if intent_type in MAIN_INTENTS:
        last_main_intent_by_session[sk] = intent_type

    # Better logging: show both detected vs resolved
    logging.info(f"intent_type(detected)..... {detected_intent}")
    logging.info(f"intent_type(resolved)..... {intent_type}")

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
        # If this is actually a generic category question, show categories instead of listing a stale category
        if ("category" in asked_now) and not ((new_entities or {}).get("category")):
            lang_root = (language or "en-IN").split("-")[0].lower()
            opts = await fetch_distinct_options(tenant_id, {}, "category", limit=12)
            reply = render_categories_reply(lang_root, opts or [])
            history.append({"role": "assistant", "content": reply})
            _commit()
            return {
                "input_text": text,
                "language": language,
                "intent_type": "asking_inquiry",
                "reply_text": reply,
                "history": history,
                "collected_entities": {},  # clear previous filters for a fresh browse
            }

        # 1) Collect and normalize entities for Pinecone filtering
        filtered_entities = filter_non_empty_entities(acc_entities)
        if 'user_name' in filtered_entities:
            del filtered_entities['user_name']
        filtered_entities_norm = normalize_entities(filtered_entities)
        filtered_entities_norm = clean_entities_for_pinecone(filtered_entities_norm)

        # 2) Fetch + dedupe products
        pinecone_data = await pinecone_fetch_records(filtered_entities_norm, tenant_id)
        pinecone_data = dedupe_products(pinecone_data)

        # 👇 EARLY EXIT when no results: be friendly and suggest other products
        if not pinecone_data:
            lang_root = (language or "en-IN").split("-")[0].lower()
            # Suggest top categories so user can pivot quickly
            try:
                top_cats = await fetch_distinct_options(tenant_id, {}, "category", limit=6)
            except Exception:
                top_cats = []
            cat_suggestion = " / ".join(top_cats[:3]) if top_cats else "Saree / Kurta Sets / Blouse"

            reply_text = (
                "Sorry, no products match those filters.\n"
                f"Would you like to check another product? For example: {cat_suggestion}"
            )

            history.append({"role": "assistant", "content": reply_text})
            _commit()
            return {
                "pinecone_data": [],
                "intent_type": intent_type,
                "language": language,
                "tenant_id": tenant_id,
                "history": history,
                "collected_entities": acc_entities,
                "reply_text": reply_text,
                "media": []
            }

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
            if k in ("category", "color", "fabric", "size", "is_rental","occasion") and v not in (None, "", [], {})
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

        # >>> ADD THIS BLOCK: when no results, return friendly message with NO follow-up
        if not product_lines:
            lang_root = (language or "en-IN").split("-")[0].lower()
            # Suggest a few categories to pivot to
            try:
                cats = await fetch_distinct_options(tenant_id, {}, "category", limit=6)
            except Exception:
                cats = []

            if lang_root == "hi":
                base = "माफ़ कीजिए, इन फ़िल्टर के साथ कोई प्रोडक्ट नहीं मिला।"
                ask  = "क्या आप कोई और कैटेगरी देखना चाहेंगे?"
            elif lang_root == "gu":
                base = "માફ કરશો, આ ફિલ્ટર સાથે કોઈ પ્રોડક્ટ મળ્યાં નથી."
                ask  = "શું તમે બીજી કેટેગરી જોવા માંગો છો?"
            else:
                base = "Sorry—no products match those filters."
                ask  = "Would you like to check another category?"

            bullet_cats = ("\n" + "\n".join(f"• {c}" for c in (cats or []))) if cats else ""
            reply_text = f"{base}\n{ask}{bullet_cats}"

            history.append({"role": "assistant", "content": reply_text})
            _commit()

            # Return with NO followup when there are no products
            payload = {
                "pinecone_data": [],
                "intent_type": intent_type,
                "language": language,
                "tenant_id": tenant_id,
                "history": history,
                "collected_entities": acc_entities,
                "followup_reply": None,
                "reply_text": reply_text,
                "media": image_urls
            }
            if mode == "call":
                payload["answer"] = reply_text
            return payload
        # <<< END ADD


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
        # Language root (en/hi/gu etc.)
        lang_root = (language or "en-IN").split("-")[0].lower()

        # 0) What did the user ask this turn?
        asked_now = detect_requested_attributes(text or "")
        if (new_entities or {}).get("category") and "category" not in asked_now:
            asked_now.append("category")
        generic_category_ask = ("category" in asked_now) and not ((new_entities or {}).get("category"))

        # 1) Merge + clean entities for this turn
        acc_clean = clean_filters_for_turn(acc_entities, asked_now)
        acc_clean = apply_price_parsing_to_entities(text or "", acc_clean)

        # --- HARD RESET for a generic "what do you have (in clothes/categories)" ask
        #     Do NOT keep any prior filters such as size/occasion/is_rental/fabric/color.
        if generic_category_ask:
            acc_clean = {}  # full fresh browse (no sticky filters)

        logging.info(f"[ASKING] asked_now={asked_now}")
        logging.info(f"[ASKING] acc_entities(before)={acc_entities}")
        logging.info(f"[ASKING] acc_clean(before sticky)={acc_clean}")

        # 2) STICKY: only re-apply previous filters when it's NOT a generic category ask
        if not generic_category_ask:
            for key in ("category", "occasion", "fabric", "color", "size", "is_rental"):
                if (acc_clean.get(key) in (None, "", [], {})) and ((acc_entities or {}).get(key) not in (None, "", [], {})):
                    acc_clean[key] = acc_entities[key]

        logging.info(f"[ASKING] acc_clean(after sticky)={acc_clean}")



        # Make rent/buy intent resilient if the text explicitly mentions it this turn
        text_lc = (text or "").lower()
        rent_triggers = ("rent pe", "rent per", "on rent", "for rent", "rent chahiye",
                         "kiraye", "kiraye pe", "भाड़े", "भाड़े", "भाड़े पर", "भाडे", "hire")
        sale_triggers = ("buy", "purchase", "khareed", "खरीद", "खरीदना")
        if acc_clean.get("is_rental") in (None, "", []):
            if any(t in text_lc for t in rent_triggers):
                acc_clean["is_rental"] = True
            elif any(t in text_lc for t in sale_triggers):
                acc_clean["is_rental"] = False

        logging.info(f"[ASKING] acc_clean(after sticky)={acc_clean}")

        # 3) EARLY EXIT: 'starting price' queries (must run BEFORE generic listing)
        if acc_clean.get("__starting_price__"):
            # For true "starting", ignore size/color/fabric to avoid inflating minimums
            local_sp = dict(acc_clean)
            for k in ("size", "color", "fabric","occasion"):
                local_sp.pop(k, None)

            price_mode = _price_mode_from_text_and_entities(text, local_sp)  # 'price' or 'rental_price'
            sp = await fetch_starting_price(tenant_id, local_sp, price_mode)

            if isinstance(sp, dict):
                reply = render_starting_price_single(lang_root, sp["category"], sp["value"], price_mode)
            else:
                reply = render_starting_price_table(lang_root, sp, price_mode)

            # persist cleaned state and return
            acc_entities.pop("__starting_price__", None)
            local_sp.pop("__starting_price__", None)
            history.append({"role": "assistant", "content": reply})
            _commit()
            return {
                "input_text": text,
                "language": language,
                "intent_type": intent_type,
                "reply_text": reply if mode != "call" else None,
                "answer": reply if mode == "call" else None,
                "history": history,
                "collected_entities": local_sp
            }

        # 4) PRODUCT LISTING when we already know the category
        if acc_clean.get("category"):
            # Drop private keys like "__starting_price__" before Pinecone
            filtered_entities = {k: v for k, v in filter_non_empty_entities(acc_clean).items() if not str(k).startswith("__")}
            filtered_entities.pop('user_name', None)

            filtered_entities_norm = normalize_entities(filtered_entities)
            filtered_entities_norm = clean_entities_for_pinecone(filtered_entities_norm)

            pinecone_data = await pinecone_fetch_records(filtered_entities_norm, tenant_id)

            # Collect up to 4 unique images
            seen, image_urls = set(), []
            for p in (pinecone_data or []):
                for u in (p.get("image_urls") or []):
                    if u and u not in seen:
                        seen.add(u)
                        image_urls.append(u)
            image_urls = image_urls[:4]

            # Build heading + lines based ONLY on collected entities so far
            collected_for_text = {
                k: v for k, v in (filtered_entities or {}).items()
                if k in ("category", "is_rental", "color", "fabric", "size")
            }

            # If no products found, graceful fallback
            if not pinecone_data:
                # Offer the user to try changing color/fabric/size or check other categories
                # Optionally list available options for one missing attribute
                try_attr = None
                for a in ("fabric", "color", "size"):
                    if not acc_clean.get(a):
                        try_attr = a
                        break
                if try_attr:
                    opts = await fetch_distinct_options(tenant_id, acc_clean, try_attr, limit=12)
                    if opts:
                        reply = render_options_reply(lang_root, try_attr, opts)
                    else:
                        reply = "Sorry, no products match your search so far."
                else:
                    reply = "Sorry, no products match your search so far."

                acc_entities = acc_clean
                history.append({"role": "assistant", "content": reply})
                _commit()
                return {
                    "pinecone_data": [],
                    "intent_type": "product_search",
                    "language": language,
                    "tenant_id": tenant_id,
                    "history": history,
                    "collected_entities": acc_entities,
                    "reply_text": reply,
                    "media": image_urls
                }

            # Otherwise, build a concise list message
            heading = _build_dynamic_heading(collected_for_text)
            lines = _build_product_lines(pinecone_data, collected_for_text)
            reply = f"{heading}\n" + "\n".join(lines)

            # Persist and return
            acc_entities = acc_clean
            history.append({"role": "assistant", "content": reply})
            _commit()
            return {
                "pinecone_data": pinecone_data,
                "intent_type": "product_search",
                "language": language,
                "tenant_id": tenant_id,
                "history": history,
                "collected_entities": acc_entities,
                "reply_text": reply,
                "media": image_urls
            }

        # 5) If category not chosen yet, help the user narrow down (show categories/options)
        # Prefer to ask for one missing attribute and show top options
        ask_attr = None
        for a in ("category", "fabric", "color", "size"):
            if not acc_clean.get(a):
                ask_attr = a
                break

        if ask_attr:
            opts = await fetch_distinct_options(tenant_id, acc_clean, ask_attr, limit=12)
            if ask_attr == "category":
                reply = render_categories_reply(lang_root, opts or [])
            else:
                reply = render_options_reply(lang_root, ask_attr, opts or [])
        else:
            # Fallback prompt
            reply = "Tell me the category or fabric/color you’re looking for."

        acc_entities = acc_clean
        history.append({"role": "assistant", "content": reply})
        _commit()
        return {
            "input_text": text,
            "language": language,
            "intent_type": intent_type,
            "reply_text": reply if mode != "call" else None,
            "answer": reply if mode == "call" else None,
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
