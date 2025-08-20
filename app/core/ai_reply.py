from __future__ import annotations
from dotenv import load_dotenv
import os
import re
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from dateutil import parser as dateparser
from app.core.central_system_prompt import Textile_Prompt 
import random
from sqlalchemy import text as sql_text  # Import if not already present
from sqlalchemy import select  # ✅
from app.db.session import SessionLocal
from app.core.rental_utils import is_variant_available
from app.core.product_search import pinecone_fetch_records
from app.core.phase_ask_inquiry import format_inquiry_reply,fetch_attribute_values,resolve_categories
from app.core.asked_now_detector import detect_requested_attributes_async
from app.db.models import Product, ProductVariant  # ✅
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

# =============== Website inquiry ===========

# --- price + formatting helpers ---
def _get_currency_symbol(p: dict) -> str:
    curr = (p.get("currency") or "INR").upper()
    return {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}.get(curr, "")

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _fmt_money(val, sym: str) -> str:
    v = _to_float(val)
    if v is None:
        return str(val) if val is not None else ""
    num = f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{sym}{num}" if sym else num

def _extract_price_fields(p: dict):
    """Return numeric values when possible for price, mrp, min_price, max_price."""
    price = next((p.get(k) for k in
                  ["sale_price", "price", "variant_price", "selling_price", "current_price"]
                  if p.get(k) not in (None, "")), None)
    mrp   = next((p.get(k) for k in
                  ["mrp", "list_price", "compare_at_price"]
                  if p.get(k) not in (None, "")), None)
    min_price = next((p.get(k) for k in ["min_price", "price_min"] if p.get(k) not in (None, "")), None)
    max_price = next((p.get(k) for k in ["max_price", "price_max"] if p.get(k) not in (None, "")), None)
    return _to_float(price), _to_float(mrp), _to_float(min_price), _to_float(max_price)

def _format_price_line(p: dict) -> str:
    """
    Pretty price line for WhatsApp:
    - "₹1,799" (only price)
    - "₹1,799 (MRP ₹2,199)" (price + mrp)
    - "₹1,799–₹2,499" (range)
    - "—" when nothing found
    """
    sym = _get_currency_symbol(p)
    price, mrp, pmin, pmax = _extract_price_fields(p)

    # Range takes priority if min/max present and meaningful
    if pmin is not None and pmax is not None and pmax >= pmin:
        return f"{_fmt_money(pmin, sym)}–{_fmt_money(pmax, sym)}"

    # Price + MRP (typical sale formatting)
    if price is not None and mrp is not None and mrp > price:
        return f"{_fmt_money(price, sym)} (MRP {_fmt_money(mrp, sym)})"

    # Single values
    if price is not None:
        return _fmt_money(price, sym)
    if mrp is not None:
        return f"MRP {_fmt_money(mrp, sym)}"
    if pmin is not None:
        return _fmt_money(pmin, sym)
    if pmax is not None:
        return _fmt_money(pmax, sym)

    return "—"


def _format_one_product_compact(p: dict) -> str:
    name   = p.get("name") or "Unnamed Product"
    color  = p.get("color")
    fabric = p.get("fabric")
    size   = p.get("size")
    tags   = " / ".join([x for x in [color, fabric, size] if x])

    price_line = _format_price_line(p)

    url = p.get("product_url") or p.get("image_url") or ""
    url_ln = f"\npreview: {url}" if url else ""

    # Exactly 4 lines (no numbering, lowercase 'preview:')
    return f"{name}\n{tags}\nPrice: {price_line}{url_ln}"


def _format_compact_products_reply(products: list[dict], max_items: int = 1) -> str:
    if not products:
        return "No items matched those filters."
    lines = [_format_one_product_compact(p) for p in products[:max_items]]
    return "\n\n".join(lines)


def _build_header(filters: dict, count: int) -> str:
    parts = []
    for k in ("category", "color", "fabric", "size"):
        v = (filters or {}).get(k)
        if v:
            parts.append(str(v).title())
    spec = " • ".join(parts)
    found = f"Found {count} match" + ("" if count == 1 else "es")
    return f"{found}{f' for {spec}' if spec else ''}:"

def _format_one_product(p: dict, idx: int) -> str:
    name   = p.get("name") or "Unnamed Product"
    color  = p.get("color")
    fabric = p.get("fabric")
    size   = p.get("size")
    tags   = " / ".join([x for x in [color, fabric, size] if x])

    price_line = _format_price_line(p)

    url = p.get("product_url") or p.get("image_url") or ""
    url_ln = f"\n   View: {url}" if url else ""

    # No rental text here
    return f"{idx}) {name}\n   {tags}\n   Price: {price_line}{url_ln}"


def format_products_reply(products: list[dict], filters: dict, max_items: int = 4) -> str:
    if not products:
        hdr = _build_header(filters, 0).replace("Found 0 matches", "No matches")
        return f"{hdr}\nNo items matched those filters.\nWould you like to see similar items or a different color/size?"

    header = _build_header(filters, len(products))
    lines = [_format_one_product(p, i + 1) for i, p in enumerate(products[:max_items])]
    extra = len(products) - max_items
    more  = f"\n\n+{extra} more. Reply with the number (e.g., 1/2/3) for details." if extra > 0 else ""
    return f"{header}\n\n" + "\n\n".join(lines) + more

FOLLOWUP_ADDRESS_TEMPLATE = (
    "Hello,\n\n"
    "Please send your First name, Last name and Full Proper address of delivery with pin-code to confirm your order.\n\n"
    "Name :_\n"
    "Contact Number :_\n"
    "Full Address :_\n"
    "1. 7 Days Return and Replacement Policy\n"
    "2. Shipping Time 5 to 7 days.\n"
    "3. Prepaid is Compulsory"
)


def choose_followup(products: list[dict], filters: dict) -> str:
    if not products:
        return "Want me to show similar designs or other variants?"
    if len(products) == 1:
        return FOLLOWUP_ADDRESS_TEMPLATE
    return "Reply with the item number (1/2/3/…) for more details. To place an order, share your delivery details in the next message."


# =========== website inquiry End ===========

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
    if size_val and size_val.lower() != "freesize":
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

def _normalize_url(url: Optional[str]) -> Optional[str]:
    """
    Ensure a usable http(s) link. Accepts bare domain or /path URLs too.
    """
    if not url or not isinstance(url, str):
        return None
    u = url.strip()
    if not u:
        return None
    if u.startswith(("http://", "https://")):
        return u
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("/"):
        # If you have a SITE_BASE env, prepend it. Otherwise return as-is.
        base = os.getenv("SITE_BASE", "").rstrip("/")
        return (base + u) if base else ("https://" + u.lstrip("/"))
    # bare domain or slug
    return "https://" + u

def _product_link_from_model(prod: Any) -> Optional[str]:
    """
    Try common attribute names on Product to get a URL.
    """
    for attr in ("product_url", "url", "page_url", "link", "permalink", "web_url"):
        val = getattr(prod, attr, None)
        if val:
            n = _normalize_url(str(val))
            if n:
                return n
    return None

def _ack_line(prod: Any) -> str:
    """
    Build the acknowledgement line with product name + link.
    """
    name = getattr(prod, "name", "") or "Selected product"
    link = _product_link_from_model(prod)
    return f"✅ Samjha: ‘{name}’" + (f" — {link}" if link else "")


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
    # ✅ clean and simple
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
        "quantity","start_date","end_date"
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
        "size", "color", "category",
        "quantity","start_date","end_date"
    ]
    field_display_names = {
        "is_rental": "rental",
        "occasion": "occasion",
        "fabric": "fabric",
        "size": "size",
        "color": "color",
        "category": "category",
        "quantity": "quantity",
        "start_date":"start_date",
        "end_date":"end_date"
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
                new_entities[k] = v.lower().strip()
            else:
                new_entities[k] = v.lower().replace(" ", "").strip()
        else:
            new_entities[k] = v
    return new_entities

# --- helper: collapse variants so every product appears once ---
def dedupe_products(pinecone_data):
    grouped = {}
    for p in (pinecone_data or []):
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
        "business_prompt": Textile_Prompt,
        "user_message": text,
        "entities_collected": {k: v for k, v in (acc_entities or {}).items() if v not in [None, "", [], {}] and k != 'user_name'},
        "recent_history": recent_history,
        "categories": categories[:12],
        "known_profile": {"name": (acc_entities or {}).get("user_name") or ""},
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
                {"role": "system", "content": "Remember: respond with JSON only — a single JSON object."}
            ],
        )
        data = json.loads(completion.choices[0].message.content)
    except Exception as e:
        logging.warning(f"LLM router JSON parse/error: {e}")
        lr = (language or "en-IN").split("-")[0].lower()
        fallback = render_categories_reply(lr, categories or ["Sarees","Kurta Sets","Lehengas","Blouses","Gowns"])
        return {"action": "help", "reply": fallback, "ask_fields": [], "confidence": 0.4}

    data.setdefault("action", "unknown")
    data.setdefault("reply", "")
    data.setdefault("ask_fields", [])
    data.setdefault("confidence", 0.6)
    if isinstance(data.get("ask_fields"), list) and len(data["ask_fields"]) > 3:
        data["ask_fields"] = data["ask_fields"][:3]

    return data

# Extract user's name (optional)
async def extract_user_name(text: str, language: Optional[str]) -> Optional[str]:
    if not text or len(text.strip()) < 2:
        return None

    sys_msg = (
        "You are a precise NER helper.\n"
        "Identify if the user is stating THEIR OWN NAME in the message.\n"
        "Only return JSON (json) with keys: has_name (bool), name (string).\n"
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
                {"role":"system","content":"Respond with JSON only — a single JSON object."}
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        if data.get("has_name") and isinstance(data.get("name"), str) and data["name"].strip():
            name = " ".join(data["name"].strip().split())
            if len(name) <= 60:
                return name
    except Exception as e:
        logging.warning(f"extract_user_name failed: {e}")
    return None


async def handle_asking_inquiry_variants(
    text: str,
    acc_entities: Dict[str, Any],
    db: AsyncSession,
    tenant_id: int,
    detect_requested_attributes_async,
) -> str:
    try:
        asked_now = await detect_requested_attributes_async(text or "", acc_entities or {})
        print("asked_now=",asked_now)
    except Exception:
        asked_now = []
        
    if not asked_now:
        asked_now = ["category"]

    needs_category = any(k in asked_now for k in ("price", "rental_price")) and not (acc_entities or {}).get("category")
    if needs_category:
        try:
            cats = await resolve_categories(db, tenant_id, {})
        except Exception:
            cats = []
        if cats:
            bullets = "\n".join(f"• {c}" for c in cats[:12])
            return f"Please choose a category for the price range:\n{bullets}"
        return "Please tell me the category for the price range."

    values = await fetch_attribute_values(db, tenant_id, asked_now, acc_entities or {})
    return format_inquiry_reply(values,acc_entities)

def merge_entities(acc_entities: Optional[Dict[str, Any]], new_entities: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge new non-empty values into accumulated entities WITHOUT wiping with None/empty.
    """
    acc = dict(acc_entities or {})
    if not new_entities:
        return acc
    for k, v in new_entities.items():
        if v not in (None, "", [], {}):
            acc[k] = v
    return acc

def _lc(x): return (str(x or "").strip().lower())

def filter_non_empty_entities(entities: dict) -> dict:
    """
    Drop keys where the value is None, empty string, or empty list/dict.
    Keeps only meaningful extracted values.
    """
    if not entities:
        return {}
    return {k: v for k, v in entities.items() if v not in (None, "", [], {})}


# ======================
# MAIN ORCHESTRATOR
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
    session_key: Optional[str] = None,
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

    # Per-customer session key
    sk = session_key or str(tenant_id)

    # Load state
    history = session_memory.get(sk, [])
    acc_entities = session_entities.get(sk, {})   # use {} not None
    last_main_intent = last_main_intent_by_session.get(sk, None)

    # --- Clean and merge new entities into memory (critical!) ---
    raw_new_entities = new_entities or {}
    clean_new_entities = filter_non_empty_entities(raw_new_entities)
    acc_entities = merge_entities(acc_entities, clean_new_entities)

    # Helpful debug logs: raw vs clean vs merged
    logging.info(f"NLU raw entities (this turn): {raw_new_entities}")
    logging.info(f"NLU clean entities (this turn): {clean_new_entities}")
    logging.info(f"Collected entities AFTER MERGE: {acc_entities}")
    
    # Reset dependent filters if category changed
    new_cat = (new_entities or {}).get("category")
    if new_cat:
        prev_cat = acc_entities.get("category")
        if not prev_cat or _lc(prev_cat) != _lc(new_cat):
            for dep in ("size", "color", "fabric", "occasion", "price", "rental_price"):
                acc_entities.pop(dep, None)
            acc_entities["category"] = new_cat
            
    def _commit():
        session_memory[sk] = history
        session_entities[sk] = acc_entities
            
            
    logging.info(f"intent_type(detected)..... {detected_intent}")
    logging.info(f"intent_type(resolved)..... {intent_type}")

    # --- greeting
    if intent_type == "greeting":
        reply = "Hello! How can I assist you today?"
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

    # --- search results
    elif intent_type == "product_search":
        turn_filters = {
            k: v for k, v in (locals().get("clean_new_entities") or {}).items()
            if v not in (None, "", [], {}) and k in ("category","color","fabric","size","is_rental","occasion")
        }

        # Fallback from memory for common facets if missing this turn
        for k in ("category", "is_rental", "color", "fabric", "size", "occasion"):
            if k not in turn_filters and acc_entities.get(k) not in (None, "", [], {}):
                turn_filters[k] = acc_entities[k]

        # Always work with a dict
        if not isinstance(turn_filters, dict):
            turn_filters = {}

        # ---- Defensive lowercase for size/category (never crash) ----
        _size_val = turn_filters.get("size")
        _cat_val  = turn_filters.get("category")

        # Normalize to strings safely
        sz  = str(_size_val or "").strip().lower()
        cat = str(_cat_val  or "").strip().lower()

        # Saree rule: Free size is meaningless for non-saree categories
        if sz == "freesize" and cat not in ("saree", "sari"):
            turn_filters.pop("size", None)

        # Now proceed with your existing normalization/search
        filtered_entities       = filter_non_empty_entities(turn_filters)
        filtered_entities_norm  = normalize_entities(filtered_entities)
        filtered_entities_norm  = clean_entities_for_pinecone(filtered_entities_norm)

        pinecone_data = await pinecone_fetch_records(filtered_entities_norm, tenant_id)
        pinecone_data = dedupe_products(pinecone_data)

        # Collect images (unchanged)
        seen, image_urls = set(), []
        for p in (pinecone_data or []):
            for u in (p.get("image_urls") or []):
                if u and u not in seen:
                    seen.add(u)
                    image_urls.append(u)
        image_urls = image_urls[:4]

        # Build text heading from what we actually showed
        collected_for_text = {
            k: v for k, v in (filtered_entities or {}).items()
            if k in ("category", "color", "fabric", "size", "is_rental", "occasion") and v not in (None, "", [], {})
        }
        heading = _build_dynamic_heading(collected_for_text)

        product_lines = []
        for product in (pinecone_data or []):
            name = product.get("name") or product.get("product_name") or "Unnamed Product"
            tags = _build_item_tags(product, collected_for_text)
            url = product.get("product_url")
            if isinstance(url, str):
                url = _normalize_url(url)
            product_lines.append(f"- {name} {tags}" + (f" — {url}" if url else ""))

        products_text = (
            f"{heading}\n" + "\n".join(product_lines)
            if product_lines else
            "Sorry, no products match your search so far."
        )

        # Use the merged memory (acc_entities) for follow-up planner
        followup = await FollowUP_Question(intent_type, acc_entities, language, session_history=history)
        reply_text = f"{products_text}"

        if mode == "call":
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
                "answer": voice_response,
                "followup_reply": followup,
                "reply_text": reply_text,
                "media": image_urls
            }
        elif mode == "chat":
            history.append({"role": "assistant", "content": reply_text})
            _commit()
            print("="*20); print(reply_text); print("="*20)
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

    # --- ✅ DIRECT PRODUCT PICK with ACK (name + link) + separate follow-ups
    # elif intent_type == "direct_product_pick":
    #     async with SessionLocal() as session_db:
    #         acc_entities = acc_entities or {}

    #         # Prefer product_id from webhook
    #         pid = (new_entities or {}).get("product_id") or acc_entities.get("product_id")

    #         # Fallback: resolve by name if provided
    #         if not pid:
    #             name = (new_entities or {}).get("product_name")
    #             if name:
    #                 row = (await session_db.execute(
    #                     select(Product.id).where(Product.name.ilike(name))
    #                 )).first()
    #                 pid = row[0] if row else None

    #         if not pid:
    #             reply = "Item exact match nahi mila. Kya similar options dikhaun?"
    #             history.append({"role": "assistant", "content": reply}); _commit()
    #             return {
    #                 "input_text": text, "language": language, "intent_type": intent_type,
    #                 "reply_text": reply, "history": history, "collected_entities": acc_entities
    #             }

    #         prod = (await session_db.execute(select(Product).where(Product.id == pid))).scalar_one_or_none()
    #         if not prod:
    #             reply = "Product not found. Similar options dikha du?"
    #             history.append({"role": "assistant", "content": reply}); _commit()
    #             return {
    #                 "input_text": text, "language": language, "intent_type": intent_type,
    #                 "reply_text": reply, "history": history, "collected_entities": acc_entities
    #             }

    #         variants = (await session_db.execute(
    #             select(ProductVariant).where(ProductVariant.product_id == pid)
    #         )).scalars().all()

    #         acc_entities["product_id"] = pid
    #         if getattr(prod, "type", None) and not acc_entities.get("category"):
    #             acc_entities["category"] = prod.type

    #         # ACK line with product name + link
    #         ack = _ack_line(prod)

    #         # Sets for choices
    #         mode_set   = {("rent" if getattr(v, "is_rental", False) else "buy") for v in variants}
    #         color_set  = {getattr(v, "color", None) for v in variants if getattr(v, "color", None)}
    #         size_set   = {getattr(v, "size", None) for v in variants if getattr(v, "size", None)}

    #         # User choices so far
    #         is_rental  = acc_entities.get("is_rental")    # True/False/None
    #         color      = acc_entities.get("color")
    #         size       = acc_entities.get("size")

    #         # Saree rule → Freesize
    #         cat = (acc_entities.get("category") or "").lower()
    #         if cat in ("saree", "sari"):
    #             acc_entities["size"] = "Freesize"
    #             size = "Freesize"

    #         # Helper: return primary + optional follow-up (for 2nd message)
    #         def _two_step(prim: str, follow: Optional[str] = None):
    #             history.append({"role": "assistant", "content": prim})
    #             _commit()
    #             payload = {
    #                 "input_text": text, "language": language, "intent_type": intent_type,
    #                 "reply_text": prim, "history": history, "collected_entities": acc_entities
    #             }
    #             if follow:
    #                 payload["followup_reply"] = follow
    #             if mode == "call":
    #                 payload["answer"] = prim if not follow else f"{prim} {follow}"
    #             return payload

    #         # 1) Ask BUY vs RENT (if both exist and not chosen)
    #         if is_rental is None and (mode_set != {"buy"}):
    #             return _two_step(ack, "Aap **Buy** karna chahenge ya **Rent**?")

    #         # 2) Ask color (if multiple colors)
    #         if not color and len(color_set) > 1:
    #             options = ", ".join(sorted(list(color_set))[:6])
    #             return _two_step(ack, f"Kaunsa color chahiye: {options}")

    #         # 3) Ask size (if needed and multiple)
    #         if not size and len(size_set) > 1 and cat not in ("saree","sari"):
    #             options = ", ".join(sorted(list(size_set))[:6])
    #             return _two_step(ack, f"Size batayiye: {options}")

    #         # Filter variants by chosen facets
    #         def ok(v):
    #             c1 = (is_rental is None) or (getattr(v, "is_rental", False) == bool(is_rental))
    #             c2 = (not color) or (str(getattr(v, "color", "")).lower() == str(color).lower())
    #             c3 = (not acc_entities.get("size")) or (str(getattr(v, "size", "")).lower() == str(acc_entities["size"]).lower())
    #             return c1 and c2 and c3

    #         cands = [v for v in variants if ok(v)]

    #         # 4) If exactly one variant → show price and ask qty (as follow-up)
    #         if len(cands) == 1:
    #             v = cands[0]
    #             price_txt = (
    #                 f"Rent ₹{int(v.rental_price)}/day" if getattr(v, "is_rental", False)
    #                 else f"Price ₹{int(v.price) if v.price is not None else 0}"
    #             )
    #             acc_entities["product_variant_id"] = v.id
    #             prim = f"{ack}\n{getattr(v,'color','')} | {getattr(v,'size','')} | {getattr(v,'fabric','')} — {price_txt}."
    #             return _two_step(prim, "Qty batayiye?")

    #         # 5) Else show 2–3 options and ask which one
    #         sample = cands[:3] if len(cands) > 3 else cands
    #         lines = []
    #         for v in sample:
    #             price_txt = (
    #                 f"Rent ₹{int(v.rental_price)}/day" if getattr(v, "is_rental", False)
    #                 else f"Price ₹{int(v.price) if v.price is not None else 0}"
    #             )
    #             color_txt = getattr(v, "color", "") or "-"
    #             size_txt  = getattr(v, "size", "") or "-"
    #             fabric_txt= getattr(v, "fabric", "") or "-"
    #             lines.append(f"- {color_txt} | {size_txt} | {fabric_txt} — {price_txt}")
    #         prim = f"{ack}\nYeh options available hain:\n" + "\n".join(lines)
    #         return _two_step(prim, "Konsa chahiye?")

    # --- availability
    elif intent_type == "availability_check":
        start_date = None
        end_date = None

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
            end_date = start_date

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

        async with SessionLocal() as db:
            available = await is_variant_available(db, int(variant_id), start_date, end_date)

        reply = f"✅ Available on {start_date.strftime('%d %b %Y')}." if available else f"❌ Not available on {start_date.strftime('%d %b %Y')}."
        history.append({"role": "assistant", "content": reply})
        _commit()
        if mode == "call":
            return {
                "input_text": text,
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
    elif intent_type == "website_inquiry":
        print("="*20)
        print(new_entities)
        print("="*20)

        filtered_entities = filter_non_empty_entities(new_entities)
        print("="*20)
        print("Filtered Entities :", filtered_entities)
        print("="*20)

        pinecone_filtered = clean_entities_for_pinecone(filtered_entities)
        print("="*20)
        print("Pinecone Filteered :", pinecone_filtered)
        print("="*20)

        print("GO for Pinecone search==========")
        pinecone_data = await pinecone_fetch_records(pinecone_filtered, tenant_id)
        print("pinecone data :", pinecone_data)
        pinecone_data = dedupe_products(pinecone_data)

        # ✅ Nicely formatted WhatsApp text (no rental text, includes price if present)
        reply_text = _format_compact_products_reply(pinecone_data, max_items=4)
        followup   = choose_followup(pinecone_data, pinecone_filtered)

        if mode == "chat":
            # Store the human-readable string in history (previously you were storing a list)
            history.append({"role": "assistant", "content": reply_text})
            _commit()
            print("="*20); print(reply_text); print("="*20)
            return {
                "pinecone_data": pinecone_data,
                "intent_type": intent_type,
                "language": language,
                "tenant_id": tenant_id,
                "history": history,
                "collected_entities": acc_entities,
                "followup_reply": followup,
                "reply_text": reply_text,
            }
        else:
            return {
                "pinecone_data": pinecone_data,
                "intent_type": intent_type,
                "language": language,
                "tenant_id": tenant_id,
                "history": history,
                "collected_entities": acc_entities,
                "followup_reply": followup,
                "reply_text": reply_text,
            }
    # --- attribute inquiry
    elif intent_type == "asking_inquiry":
        async with SessionLocal() as session:
            reply_text = await handle_asking_inquiry_variants(
                text=text,
                acc_entities=acc_entities or {},
                db=session,
                tenant_id=tenant_id,
                detect_requested_attributes_async=detect_requested_attributes_async,
            )
            print("reply_text=",reply_text)
        history.append({"role": "assistant", "content": reply_text})
        _commit()

        payload = {
            "input_text": text,
            "language": language,
            "intent_type": "asking_inquiry",
            "history": history,
            "collected_entities": acc_entities,
        }
        if mode == "call":
            payload["answer"] = reply_text
        else:
            payload["reply_text"] = reply_text

        return payload
       
    # --- other / fallback
    elif intent_type == "other" or intent_type is None:
        routed = await llm_route_other(text, language, tenant_id, acc_entities, history)
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
                "answer": reply,
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
    else:
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
