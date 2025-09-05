# app/api/whatsapp.py

from fastapi import FastAPI, Request, Response, APIRouter
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os
import json
import logging
import httpx
import asyncio
from app.db.session import get_db
from app.core.lang_utils import detect_language
from app.core.intent_utils import detect_textile_intent_openai
from app.core.ai_reply import analyze_message
from app.core.chat_persistence import (
    get_or_create_customer,
    get_or_open_active_session,
    append_transcript_message,
)

from app.services.wa_media import download_media_bytes
from app.services.wa_media import get_media_url_and_meta
from app.services.visual_search import visual_search_bytes_sync, format_matches_for_whatsapp_images,group_matches_by_product
from app.core.product_pick_ import (
    resolve_product_from_caption_async,
    get_attrs_for_product_async,
    extract_attrs_from_text,find_assistant_image_caption_by_msg_id,find_prev_assistant_image_caption,_ensure_allowed_lists
)

from uuid import uuid4
# ADD VTO IMPORT
from app.core.virtual_try_on import generate_vto_image, VTOConfig
import re
from sqlalchemy import text
from app.db.session import SessionLocal # ensure this is imported

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

logging.info("Conversation language remains as")

WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
META_APP_SECRET = os.getenv("META_APP_SECRET")

router = APIRouter()

# Simple in-memory deduplication for processed incoming message SIDs
processed_message_sids = {}

mode = "chat"

# VTO STATE MANAGEMENT - KEEP EXISTING
_vto_state: dict[str, dict] = {}

# ADD NEW VTO STATE FUNCTIONS
def _get_vto_state(session_key: str) -> dict:
    """Get VTO state for a session"""
    return _vto_state.get(session_key, {})

def _set_vto_state(session_key: str, state: dict):
    """Set VTO state for a session"""
    _vto_state[session_key] = state

def _clear_vto_state(session_key: str):
    """Clear VTO state for a session"""
    _vto_state.pop(session_key, None)

def _is_vto_flow_active(session_key: str) -> bool:
    """Check if VTO flow is active for this session"""
    state = _get_vto_state(session_key)
    return state.get("active", False)



async def get_tenant_id_by_phone(phone_number: str, db):
    """
    Fetch tenant id by phone number from the database.
    """
    phone_number = _normalize_business_number(phone_number)
    query = text("SELECT id FROM tenants WHERE whatsapp_number = :phone AND is_active = true LIMIT 1")
    result = await db.execute(query, {"phone": phone_number})
    row = result.fetchone()
    if row:
        return row[0]
    return None

async def update_customer_language(db, customer_id: int, language: str):
    """
    Update the preferred_language for a customer in the database.
    """
    query = text("UPDATE customers SET preferred_language = :language WHERE id = :customer_id")
    await db.execute(query, {"language": language, "customer_id": customer_id})
    # No commit here; it's handled in the main flow

async def get_tenant_name_by_phone(phone_number: str, db):
    query = text("SELECT name FROM tenants WHERE whatsapp_number = :phone AND is_active = true LIMIT 1")
    result = await db.execute(query, {"phone": phone_number})
    row = result.fetchone()
    return row[0] if row else None

#fabric by tenant_id
async def get_tenant_fabric_by_phone(phone_number: str, db):
    # 1) Find tenant id for this WhatsApp number
    tid = (await db.execute(
        text("""SELECT id FROM tenants
                WHERE whatsapp_number = :phone AND is_active = true
                LIMIT 1"""),
        {"phone": phone_number}
    )).scalar()

    if not tid:
        return []

    # 2) Helper: does table.column exist?
    async def col_exists(table: str, col: str) -> bool:
        q = text("""
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = :table
            AND column_name = :col
            AND table_schema = ANY (current_schemas(false))
            LIMIT 1
        """)
        return (await db.execute(q, {"table": table, "col": col})).first() is not None

    sources = []

    # product_variants.fabric (if present)
    if await col_exists("product_variants", "fabric"):
        sources.append("""
            SELECT LOWER(TRIM(pv.fabric)) AS fab
            FROM product_variants pv
            JOIN products p ON p.id = pv.product_id
            WHERE p.tenant_id = :tid
            AND pv.fabric IS NOT NULL
            AND TRIM(pv.fabric) <> ''
        """)

    if not sources:
        return []

    union_sql = " UNION ALL ".join(sources)
    final_sql = f"SELECT DISTINCT fab FROM ({union_sql}) s"
    result = await db.execute(text(final_sql), {"tid": tid})
    rows = result.fetchall() or []
    return [r[0] for r in rows]

#color by tenant_id
async def get_tenant_color_by_phone(phone_number: str, db):
    # 1) Find tenant id for this WhatsApp number
    tid = (await db.execute(
        text("""SELECT id FROM tenants
                WHERE whatsapp_number = :phone AND is_active = true
                LIMIT 1"""),
        {"phone": phone_number}
    )).scalar()

    if not tid:
        return []

    # 2) Helper: does table.column exist?
    async def col_exists(table: str, col: str) -> bool:
        q = text("""
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = :table
            AND column_name = :col
            AND table_schema = ANY (current_schemas(false))
            LIMIT 1
        """)
        return (await db.execute(q, {"table": table, "col": col})).first() is not None

    sources = []

    # product_variants.fabric (if present)
    if await col_exists("product_variants", "color"):
        sources.append("""
            SELECT LOWER(TRIM(pv.color)) AS fab
            FROM product_variants pv
            JOIN products p ON p.id = pv.product_id
            WHERE p.tenant_id = :tid
            AND pv.color IS NOT NULL
            AND TRIM(pv.color) <> ''
        """)

    if not sources:
        return []

    union_sql = " UNION ALL ".join(sources)
    final_sql = f"SELECT DISTINCT fab FROM ({union_sql}) s"
    result = await db.execute(text(final_sql), {"tid": tid})
    rows = result.fetchall() or []
    return [r[0] for r in rows]

#size by tenant_id
async def get_tenant_size_by_phone(phone_number: str, db):
    # 1) Find tenant id for this WhatsApp number
    tid = (await db.execute(
        text("""SELECT id FROM tenants
                WHERE whatsapp_number = :phone AND is_active = true
                LIMIT 1"""),
        {"phone": phone_number}
    )).scalar()

    if not tid:
        return []

    # 2) Helper: does table.column exist?
    async def col_exists(table: str, col: str) -> bool:
        q = text("""
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = :table
            AND column_name = :col
            AND table_schema = ANY (current_schemas(false))
            LIMIT 1
        """)
        return (await db.execute(q, {"table": table, "col": col})).first() is not None

    sources = []

    # product_variants.fabric (if present)
    if await col_exists("product_variants", "size"):
        sources.append("""
            SELECT LOWER(TRIM(pv.size)) AS fab
            FROM product_variants pv
            JOIN products p ON p.id = pv.product_id
            WHERE p.tenant_id = :tid
            AND pv.size IS NOT NULL
            AND TRIM(pv.size) <> ''
        """)

    if not sources:
        return []

    union_sql = " UNION ALL ".join(sources)
    final_sql = f"SELECT DISTINCT fab FROM ({union_sql}) s"
    result = await db.execute(text(final_sql), {"tid": tid})
    rows = result.fetchall() or []
    return [r[0] for r in rows]

#category by tenant_id
async def get_tenant_category_by_phone(phone_number: str, db):
    """
    Return a lowercase, de-duplicated list of category names available for the tenant
    mapped to this WhatsApp number. Pulls from products.category and products.type.
    """
    query = text("""
        WITH t AS (
            SELECT id
            FROM tenants
            WHERE whatsapp_number = :phone AND is_active = true
            LIMIT 1
        )
        -- pull from products.category
        SELECT DISTINCT LOWER(TRIM(p.category)) AS cat
        FROM products p, t
        WHERE p.tenant_id = t.id
        AND p.category IS NOT NULL
        AND TRIM(p.category) <> '';
    """)
    result = await db.execute(query, {"phone": phone_number})
    rows = result.fetchall() or []
    # return a simple Python list of strings
    return [r[0] for r in rows]

#type by tenant_id
async def get_tenant_type_by_phone(phone_number: str, db):
    """
    Return a lowercase, de-duplicated list of category names available for the tenant
    mapped to this WhatsApp number. Pulls from products.category and products.type.
    """
    query = text("""
        WITH t AS (
            SELECT id
            FROM tenants
            WHERE whatsapp_number = :phone AND is_active = true
            LIMIT 1
        )
        -- pull from products.type
        SELECT DISTINCT LOWER(TRIM(p.type)) AS cat
        FROM products p, t
        WHERE p.tenant_id = t.id
        AND p.type IS NOT NULL
        AND TRIM(p.type) <> '';
    """)
    result = await db.execute(query, {"phone": phone_number})
    rows = result.fetchall() or []
    # return a simple Python list of strings
    return [r[0] for r in rows]

# occasion bby tenant_id
async def get_tenant_occasion_by_phone(phone_number: str, db):
    """
    Return a lowercase, de-duplicated list of occasion names available for the tenant
    mapped to this WhatsApp number.
    Uses:
    tenants.whatsapp_number -> tenants.id
    products.tenant_id -> products.id
    product_variants.product_id -> product_variants.id
    product_variant_occasions.variant_id -> occasions.id
    Filters out empty names and (by default) inactive variants.
    """
    query = text("""
        WITH t AS (
            SELECT id
            FROM tenants
            WHERE whatsapp_number = :phone
            AND is_active = TRUE
            ORDER BY id
            LIMIT 1
        )
        SELECT DISTINCT LOWER(TRIM(o.name)) AS occ
        FROM t
        JOIN products p
            ON p.tenant_id = t.id
        JOIN product_variants pv
            ON pv.product_id = p.id
        JOIN product_variant_occasions pvo
            ON pvo.variant_id = pv.id
        JOIN occasions o
            ON o.id = pvo.occasion_id
        WHERE COALESCE(pv.is_active, TRUE) = TRUE
        AND o.name IS NOT NULL
        AND TRIM(o.name) <> '';
    """)
    result = await db.execute(query, {"phone": phone_number})
    rows = result.fetchall() or []
    return [r[0] for r in rows]

def _save_public_png_and_get_url(png_bytes: bytes) -> str | None:
    base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    static_dir = os.getenv("PUBLIC_STATIC_DIR", "")
    if not base or not static_dir:
        return None  # You'll fall back to text if you don't configure hosting

    try:
        os.makedirs(static_dir, exist_ok=True)
        name = f"vto_{uuid4().hex}.png"
        fpath = os.path.join(static_dir, name)
        with open(fpath, "wb") as f:
            f.write(png_bytes)
        return f"{base}/static/{name}"
    except Exception:
        logging.exception("Failed to save VTO public file")
        return None

async def _get_replied_bot_image_url(db, chat_session_id: int, replied_message_id: str | None) -> str | None:
    """Return the image URL from the assistant message the user replied to."""
    if not replied_message_id:
        return None

    # If you already have helpers like below, prefer them:
    try:
        # Example: replace by your own query helpers if they exist.
        cap = await find_assistant_image_caption_by_msg_id(db, replied_message_id)  # ← your helper
        if isinstance(cap, dict):
            return cap.get("image_url") or cap.get("url")
    except Exception:
        pass

    # Fallback: last assistant image in this session (implement to match your schema)
    try:
        rec = await find_prev_assistant_image_caption(db, chat_session_id)  # ← your helper
        if isinstance(rec, dict):
            return rec.get("image_url") or rec.get("url")
    except Exception:
        pass

    return None

async def send_whatsapp_reply_cloud(
    to_waid: str,
    body,
    context_msg_id: str | None = None,
    phone_number_id: str | None = None,
) -> str | None:
    msg = body if isinstance(body, str) else (body.get("reply_text") if isinstance(body, dict) else str(body))
    pnid = phone_number_id  # fallback only if you still keep it for legacy

    if not pnid or not WHATSAPP_TOKEN:
        logging.error("Cloud API missing: phone_number_id or WHATSAPP_TOKEN")
        return None

    url = f"https://graph.facebook.com/v20.0/{pnid}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_waid.replace("+", "").strip(),
        "type": "text",
        "text": {"body": (msg or "")[:4096]},
    }

    if context_msg_id:
        payload["context"] = {"message_id": context_msg_id}

    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(url, json=payload, headers=headers)
        logging.info(f"[CLOUD] Send resp: {resp.status_code} {resp.text}")

        try:
            data = resp.json()
            return (data.get("messages") or [{}])[0].get("id")
        except Exception:
            return None

async def send_whatsapp_image_cloud(
    to_waid: str,
    image_url: str,
    caption: str | None = None,
    context_msg_id: str | None = None,
    phone_number_id: str | None = None,
) -> str | None:
    pnid = phone_number_id

    if not pnid or not WHATSAPP_TOKEN:
        logging.error("Cloud API missing: phone_number_id or WHATSAPP_TOKEN")
        return None

    url = f"https://graph.facebook.com/v20.0/{pnid}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_waid.replace("+", "").strip(),
        "type": "image",
        "image": {"link": image_url},
    }

    if caption:
        payload["image"]["caption"] = caption[:1024]

    if context_msg_id:
        payload["context"] = {"message_id": context_msg_id}

    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(url, json=payload, headers=headers)
        logging.info(f"[CLOUD] Send image resp: {resp.status_code} {resp.text}")

        try:
            data = resp.json()
            return (data.get("messages") or [{}])[0].get("id")
        except Exception:
            return None

# ------- Normalize Phone ----------

def _normalize_waid_phone(waid: str | None) -> str | None:
    """Turn '918799559020' into '+918799559020' for customers.phone."""
    if not waid:
        return None

    digits = re.sub(r"\D+", "", waid)
    return f"+{digits}" if digits else None

def _normalize_url(u: str | None) -> str | None:
    if not u:
        return None
    u = str(u).strip()
    if u.startswith(("http://", "https://")): return u
    if u.startswith("//"): return "https:" + u
    return u

# def _product_caption(prod: dict) -> str:
#     name = prod.get("name") or prod.get("product_name") or "Product"
#     tag = "Rent" if prod.get("is_rental") else "sale"
#     url = _normalize_url(prod.get("product_url") or prod.get("image_url"))
#     cap = f"{name} - {tag}" + (f" — {url}" if url else "")
#     return cap[:1024]

def _product_caption(prod: dict) -> str:
    name = prod.get("name") or prod.get("product_name") or "Product"
    url = _normalize_url(prod.get("product_url") or prod.get("image_url"))

    # helpers
    def _to_bool(v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in {"true", "1", "yes", "y"}

    def _currency_symbol():
        cur = str(prod.get("currency") or "INR").upper()
        return {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}.get(cur, "")

    def _fmt_money(val):
        if val in (None, "", "-"):
            return "—"
        try:
            n = float(str(val).replace(",", ""))
            s = f"{n:,.2f}".rstrip("0").rstrip(".")
            return f"{_currency_symbol()}{s}"
        except Exception:
            # if it's already a formatted string (e.g., "₹999"), just return as-is
            return str(val)

    is_rental = _to_bool(prod.get("is_rental"))
    if is_rental:
        label = "Rent"
        amount = _fmt_money(prod.get("rental_price"))
    else:
        label = "Sale"
        price = (prod.get("price") or prod.get("sale_price") or prod.get("selling_price")
                or prod.get("variant_price") or prod.get("current_price"))
        amount = _fmt_money(price)

    cap = f"{name}\n{label}: {amount}\n"
    if url:
        cap += f"Webiste: {url}"

    return cap[:1024]

# ---------- NEW HELPERS: direct product pick ----------

async def get_tenant_products_by_phone(phone_number: str, db):
    """
    Return list[(product_id, product_name)] for the tenant mapped to business WhatsApp number.
    Only non-empty names are returned.
    """
    q = text("""
        WITH t AS (
            SELECT id FROM tenants WHERE whatsapp_number = :phone AND is_active = true LIMIT 1
        )
        SELECT p.id, p.name
        FROM products p, t
        WHERE p.tenant_id = t.id
        AND p.name IS NOT NULL
        AND TRIM(p.name) <> ''
    """)
    rows = (await db.execute(q, {"phone": phone_number})).fetchall() or []
    return [(r[0], r[1]) for r in rows]

# ---------- Get Transcript from DB Via Whastapp Number -------------
async def get_transcript_by_phone(phone_number: str, db):
    query = text("""
        WITH c AS (
            SELECT id
            FROM customers
            WHERE phone = :phone
            AND is_active = TRUE
            LIMIT 1
        )
        SELECT cs.transcript
        FROM chat_sessions cs, c
        WHERE cs.customer_id = c.id
        ORDER BY cs.started_at DESC
        LIMIT 1
    """)
    result = await db.execute(query, {"phone": phone_number})
    result = result.scalar_one_or_none()

    if result is None:
        return []

    # If stored as text, decode; if JSONB, driver already gives list/dict
    if isinstance(result, (bytes, bytearray)):
        result = result.decode("utf-8", errors="ignore")

    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return []

    return result if isinstance(result, (list, dict)) else []

def _normalize_messages(transcript):
    """
    Returns a flat list[dict] of messages from whatever structure you stored.
    Accepts list or dict or JSON string.
    """
    if transcript is None:
        return []

    data = transcript
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return []

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Common containers you might have used
        for key in ("history", "messages", "items"):
            val = data.get(key)
            if isinstance(val, list):
                return val

        # If the whole dict is a single message, wrap it
        if {"role", "msg_id"} <= set(data.keys()):
            return [data]

        return []

    return []

def _extract_reply_to_id(msg: dict) -> str | None:
    """Get reply_to id from message meta or raw.context.id."""
    # Preferred: your flattened meta
    reply_to = (msg.get("meta") or {}).get("reply_to")
    if reply_to:
        return reply_to

    # Fallback: pull from raw Cloud API context.id if present
    try:
        return msg["meta"]["raw"]["entry"][0]["changes"][0]["value"]["messages"][0]["context"]["id"]
    except Exception:
        return None

def find_assistant_text_by_msg_id(messages: list[dict], msg_id: str) -> str | None:
    for m in messages:
        if m.get("msg_id") == msg_id and m.get("role") == "assistant":
            return m.get("text")
    return None

# --- Cloud API Webhook (Meta) ----------------------------------------------

processed_meta_msg_ids = set()

@router.get("/webhook")
async def verify_cloud_webhook(request: Request):
    """Meta verification handshake: echo hub.challenge if token matches."""
    params = request.query_params

    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge", "")

    logging.info("-------In Get Method Webhook-----")

    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        try:
            return int(challenge)
        except ValueError:
            return challenge

    return Response(status_code=403)

def _valid_signature(app_secret: str, raw: bytes, header: str) -> bool:
    if not app_secret:
        return True
    if not header or not header.startswith("sha256="):
        return False

    import hmac, hashlib
    expected = "sha256=" + hmac.new(app_secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)

def _primary_image_for_product(prod: dict) -> str | None:
    # prefer product.image_urls[0], then product.image_url
    for u in (prod.get("image_urls") or []):
        if u:
            return _normalize_url(u)
    return _normalize_url(prod.get("image_url"))

# --------FInding Entites by reply id --------------
def find_entities_by_msg_id(messages: list[dict], msg_id: str):
    """
    Return the entities saved alongside an assistant message, searching:
    1) the exact message (meta.entities)
    2) a companion row with msg_id == f"{msg_id}:entities"
       (take meta.entities if present, else parse text as JSON)
    """
    if not msg_id:
        return None

    # 1) exact message
    for m in messages:
        if m.get("msg_id") == msg_id and m.get("role") == "assistant":
            ents = (m.get("meta") or {}).get("entities")
            if ents not in (None, "", [], {}):
                return ents

    # 2) companion ':entities' row
    ent_id = f"{msg_id}:entities"
    for m in messages:
        if m.get("msg_id") == ent_id and m.get("role") == "assistant":
            ents = (m.get("meta") or {}).get("entities")
            if ents not in (None, "", [], {}):
                return ents
            # fallback: parse the text if it looks like JSON
            try:
                t = m.get("text")
                if isinstance(t, str) and t.strip().startswith("{"):
                    return json.loads(t)
            except Exception:
                pass

    return None

# -----------merge Entitie sfor reply message
def _merge_entities(base: dict | None, overlay: dict | None, override: bool = False) -> dict:
    """
    Merge overlay into base.
    - If override=False (default): only fill when base[key] is empty/None/""
    - If override=True: overlay overwrites base for matching keys
    """
    if not isinstance(base, dict):
        base = {}
    if not isinstance(overlay, dict):
        return base

    merged = dict(base)
    for k, v in overlay.items():
        if v in (None, "", [], {}):
            continue
        if override:
            merged[k] = v
        else:
            if k not in merged or merged[k] in (None, "", [], {}):
                merged[k] = v

    return merged

# ----------- Static FollowUp For Visual Search -------------
def _visual_followup_text(lang: str = "en-IN") -> str:
    m = {
        "gu-IN": "👇 તમારે ગમે તે પ્રોડક્ટ પર 'I want this' લખીને જવાબ આપો, અથવા fabric/રંગ/size કહો જેથી હું વધુ સારી રીતે બતાવી શકું.",
        "hi-IN": "👇 जो पसंद आए उस प्रोडक्ट पर 'I want this' लिखकर रिप्लाई करें, या fabric/रंग/size बताएं ताकि मैं और बेहतर दिखा सकूं।",
        "en-IN": "👇 Reply 'I want this' on the product you like, or tell me fabric/colour/size to refine.",
        "en-US": "👇 Reply 'I want this' on the product you like, or tell me fabric/color/size to refine.",
    }
    return m.get(lang, m["en-IN"])

# ---- helper: normalize out_msgs tuples ----
def _iter_out_msgs_with_meta(out_msgs):
    """
    Accepts tuples of either:
    ("text", caption, msg_id)
    ("image", caption, msg_id)
    ("image", caption, msg_id, image_url)  # preferred for images

    and always yields (kind, text, msg_id, image_url_or_None).
    """
    for item in out_msgs:
        if len(item) == 4:
            kind, txt, mid, img_url = item
        else:
            kind, txt, mid = item
            img_url = None

        yield kind, txt, mid, img_url

# Return digits-only (E.164 without +) for reliable DB matching.
def _normalize_business_number(s: str | None) -> str:
    """Return digits-only (E.164 without +) for reliable DB matching."""
    return re.sub(r"\D+", "", s or "")

async def _lookup_display_phone_number_digits(phone_number_id: str) -> str:
    """GET /{phone_number_id}?fields=display_phone_number and normalize."""
    if not WHATSAPP_TOKEN:
        return ""

    url = f"https://graph.facebook.com/v20.0/{phone_number_id}"
    params = {"fields": "display_phone_number"}
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params, headers=headers)
            data = resp.json()
            return _normalize_business_number(data.get("display_phone_number"))
    except Exception:
        logging.exception("[CLOUD] phone_number_id lookup failed")
        return ""
    

# ADD NEW VTO HELPER FUNCTIONS
def _detect_vto_keywords(text: str) -> bool:
    """Detect if user wants virtual try-on"""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # English keywords
    vto_keywords = [
        "virtual try on", "virtual try-on", "vto", "try on", "try-on",
        "how will this look on me", "how would this look", "see on me",
        "try this", "fit on me", "wear this", "looks on me"
    ]
    
    # Hindi keywords  
    hindi_keywords = [
        "वर्चुअल ट्राई ऑन", "ट्राई ऑन", "मुझ पर कैसा लगेगा", "पहनकर दिखाओ"
    ]
    
    # Gujarati keywords
    gujarati_keywords = [
        "વર્ચ્યુઅલ ટ્રાય ઓન", "ટ્રાય ઓન", "મારા પર કેવું લાગશે", "પહેરીને બતાવો"
    ]
    
    all_keywords = vto_keywords + hindi_keywords + gujarati_keywords
    
    return any(keyword in text_lower for keyword in all_keywords)

def _get_vto_messages(lang: str = "en-IN") -> dict:
    """Get localized VTO messages"""
    lr = lang.split('-')[0].lower()
    
    if lr == "hi":
        return {
            "need_person": "वर्चुअल ट्राई-ऑन के लिए कृपया अपनी एक स्पष्ट, सामने से ली गई पूरे शरीर की या कमर तक की फोटो भेजें। अच्छी रोशनी और सादे बैकग्राउंड के साथ।",
            "need_garment": "कृपया उस कपड़े की फोटो भेजें जिसे आप ट्राई करना चाहते हैं।",
            "processing": "🔄 वर्चुअल ट्राई-ऑन तैयार कर रहे हैं... कृपया थोड़ा इंतजार करें।",
            "ready": "✨ आपका वर्चुअल ट्राई-ऑन तैयार है!",
            "error": "❌ वर्चुअल ट्राई-ऑन में कोई समस्या हुई। कृपया फिर से कोशिश करें।",
            "invalid_image": "कृपया एक स्पष्ट फोटो भेजें।"
        }
    elif lr == "gu":
        return {
            "need_person": "વર્ચ્યુઅલ ટ્રાય-ઓન માટે કૃપા કરીને તમારો એક સ્પષ્ટ, આગળથી લેવાયેલો પૂરા શરીરનો અથવા કમર સુધીનો ફોટો મોકલો. સારા પ્રકાશ અને સાદા બેકગ્રાઉન્ડ સાથે.",
            "need_garment": "કૃપા કરીને તે કપડાંનો ફોટો મોકલો જેને તમે ટ્રાય કરવા માગો છો.",
            "processing": "🔄 વર્ચ્યુઅલ ટ્રાય-ઓન તૈયાર કરી રહ્યા છીએ... કૃપા કરી થોડી રાહ જુઓ.",
            "ready": "✨ તમારો વર્ચ્યુઅલ ટ્રાય-ઓન તૈયાર છે!",
            "error": "❌ વર્ચ્યુઅલ ટ્રાય-ઓનમાં કોઈ સમસ્યા થઈ. કૃપા કરી ફરીથી પ્રયાસ કરો.",
            "invalid_image": "કૃપા કરીને એક સ્પષ્ટ ફોટો મોકલો."
        }
    else:  # English
        return {
            "need_person": "For virtual try-on, please send a clear, front-facing full-body or waist-up photo of yourself with good lighting and plain background.",
            "need_garment": "Please send the photo of the garment you want to try on.",
            "processing": "🔄 Generating your virtual try-on... please wait a moment.",
            "ready": "✨ Your virtual try-on is ready!",
            "error": "❌ Something went wrong with the virtual try-on. Please try again.",
            "invalid_image": "Please send a clear photo."
        }

# --- inside whatsapp.py ---

async def _handle_vto_flow(
    session_key: str,
    text_msg: str,
    mtype: str,
    msg: dict,
    from_waid: str,
    outbound_pnid: str | None,
    current_language: str,
    db,
    chat_session,
    msg_id: str,
    request,  # required by your code
):
    """
    Returns (handled: bool, out_msgs: list[(kind, text, mid, [image_url])])
    """
    vto_messages = _get_vto_messages(current_language)
    out_msgs: list[tuple] = []

    # helpers bound to this session
    def _get(): return _get_vto_state(session_key) or {}
    def _set(s): return _set_vto_state(session_key, s)
    def _have_both(s):
        return bool(s.get("person_image")) and bool(s.get("garment_image") or s.get("garment_image_url"))

    async def _bytes_from_url(url: str) -> bytes:
        import httpx
        async with httpx.AsyncClient(timeout=30) as hc:
            r = await hc.get(url)
            r.raise_for_status()
            return r.content

    async def _resolve_garment_bytes(s) -> bytes | None:
        """Get garment bytes from in-memory bytes or the stored garment_image_url."""
        if s.get("garment_image"):
            return s["garment_image"]
        if s.get("garment_image_url"):
            try:
                return await _bytes_from_url(s["garment_image_url"])
            except Exception:
                logging.exception("[VTO] Failed to fetch garment bytes from URL")
                return None
        return None

    # --- state snapshot BEFORE handling
    vto_state = _get()
    logging.info(
        "[VTO] pre-handle (normal flow) | session=%s active=%s step=%s person=%s garment_bytes=%s garment_url=%s",
        session_key, vto_state.get("active"), vto_state.get("step"),
        "yes" if vto_state.get("person_image") else "no",
        "yes" if vto_state.get("garment_image") else "no",
        "yes" if vto_state.get("garment_image_url") else "no",
    )

    # If VTO isn't active, nothing to do
    if not vto_state.get("active"):
        return False, []

    current_step = vto_state.get("step") or "need_person"

    # If text arrives while we're waiting for an image, gently remind once
    if mtype == "text" and ("need_person" in current_step or "need_garment" in current_step):
        reminder_key = "need_person" if current_step == "need_person" else "need_garment"
        reminder = vto_messages.get(reminder_key)
        mid = await send_whatsapp_reply_cloud(
            to_waid=from_waid,
            body=reminder,
            phone_number_id=outbound_pnid,
        )
        out_msgs.append(("text", reminder, mid))
        logging.info("[VTO][REMIND] session=%s step=%s -> sent: %s", session_key, current_step, reminder_key)
        # keep state unchanged
        logging.info(
            "[VTO] post-handle (normal flow) | session=%s active=%s step=%s person=%s garment_bytes=%s garment_url=%s",
            session_key, vto_state.get("active"), vto_state.get("step"),
            "yes" if vto_state.get("person_image") else "no",
            "yes" if vto_state.get("garment_image") else "no",
            "yes" if vto_state.get("garment_image_url") else "no",
        )
        return True, out_msgs

    # ---- IMAGE upload handling
    if mtype == "image" and "image" in msg:
        try:
            # Download inbound bytes (works for real webhook & local tester)
            img_obj = msg.get("image", {}) if isinstance(msg, dict) else {}
            media_id = img_obj.get("id")
            link = img_obj.get("link")
            is_local = request.headers.get("X-LOCAL-TEST") == "1"

            if is_local and link:
                import httpx
                async with httpx.AsyncClient(timeout=30) as hc:
                    r = await hc.get(link)
                    r.raise_for_status()
                    img_bytes = r.content
            else:
                if not media_id:
                    raise RuntimeError("no-media-id")
                img_bytes = await download_media_bytes(media_id)

            if not img_bytes:
                raise RuntimeError("empty-image")

            # Persist into state based on the step we are in
            if current_step == "need_person":
                vto_state["person_image"] = img_bytes

                # IMPORTANT: if we already have a garment (bytes or URL), do NOT ask again → go generate
                if vto_state.get("garment_image") or vto_state.get("garment_image_url"):
                    logging.info(
                        "[VTO][READY] Both images present for session=%s (person=bytes; garment=%s)",
                        session_key, "bytes" if vto_state.get("garment_image") else "url"
                    )

                    # Tell user we’re generating right away
                    mid = await send_whatsapp_reply_cloud(
                        to_waid=from_waid,
                        body=vto_messages["processing"],
                        phone_number_id=outbound_pnid,
                    )
                    out_msgs.append(("text", vto_messages["processing"], mid))

                    # Run VTO
                    try:
                        garment_bytes = await _resolve_garment_bytes(vto_state)
                        result_bytes = await generate_vto_image(
                            person_bytes=vto_state["person_image"],
                            garment_bytes=garment_bytes,
                            cfg=VTOConfig(base_steps=60, add_watermark=False),
                        )
                        # Send result
                        public_url = _save_public_png_and_get_url(result_bytes)
                        logging.info("="*100)
                        logging.info(f"Public Url:{public_url}")
                        logging.info("="*100)
                        if public_url:
                            result_mid = await send_whatsapp_image_cloud(
                                to_waid=from_waid,
                                image_url=public_url,
                                caption=vto_messages["ready"],
                                phone_number_id=outbound_pnid,
                            )
                            out_msgs.append(("image", vto_messages["ready"], result_mid, public_url))
                        else:
                            # hosting not configured
                            result_mid = await send_whatsapp_reply_cloud(
                                to_waid=from_waid,
                                body=f"{vto_messages['ready']} (image hosting unavailable)",
                                phone_number_id=outbound_pnid,
                            )
                            out_msgs.append(("text", f"{vto_messages['ready']} (image hosting unavailable)", result_mid))

                        # Clear VTO session
                        _set({"active": False})
                        logging.info("[VTO][DONE] session=%s (state cleared)", session_key)

                    except Exception:
                        logging.exception("[VTO] Generation failed")
                        fail_mid = await send_whatsapp_reply_cloud(
                            to_waid=from_waid,
                            body=vto_messages["error"],
                            phone_number_id=outbound_pnid,
                        )
                        out_msgs.append(("text", vto_messages["error"], fail_mid))
                        _set({})  # reset state
                        logging.info("[VTO][RESET] session=%s due to failure", session_key)

                    logging.info(
                        "[VTO] post-handle (normal flow) | session=%s active=%s step=%s person=%s garment_bytes=%s garment_url=%s",
                        session_key, _get().get("active"), _get().get("step"),
                        "yes" if _get().get("person_image") else "no",
                        "yes" if _get().get("garment_image") else "no",
                        "yes" if _get().get("garment_image_url") else "no",
                    )
                    return True, out_msgs

                # otherwise we still need a garment photo
                vto_state["step"] = "need_garment"
                _set(vto_state)
                mid = await send_whatsapp_reply_cloud(
                    to_waid=from_waid,
                    body=vto_messages["need_garment"],
                    phone_number_id=outbound_pnid,
                )
                out_msgs.append(("text", vto_messages["need_garment"], mid))
                logging.info("[VTO][PHASE] session=%s moved -> need_garment", session_key)
                return True, out_msgs

            # If we were waiting for the garment
            elif current_step == "need_garment":
                vto_state["garment_image"] = img_bytes
                _set(vto_state)

                logging.info("[VTO][READY] Both images present for session=%s (person=%s; garment=bytes)",
                             session_key, "yes" if vto_state.get("person_image") else "no")

                # Tell user we're generating
                mid = await send_whatsapp_reply_cloud(
                    to_waid=from_waid,
                    body=vto_messages["processing"],
                    phone_number_id=outbound_pnid,
                )
                out_msgs.append(("text", vto_messages["processing"], mid))

                # Run VTO
                try:
                    person = vto_state.get("person_image")
                    garment = vto_state.get("garment_image")
                    result_bytes = await generate_vto_image(
                        person_bytes=person,
                        garment_bytes=garment,
                        cfg=VTOConfig(base_steps=60, add_watermark=False),
                    )

                    public_url = _save_public_png_and_get_url(result_bytes)
                    logging.info("="*100)
                    logging.info(f"Public Url:{public_url}")
                    logging.info("="*100)
                    if public_url:
                        result_mid = await send_whatsapp_image_cloud(
                            to_waid=from_waid,
                            image_url=public_url,
                            caption=vto_messages["ready"],
                            phone_number_id=outbound_pnid,
                        )
                        out_msgs.append(("image", vto_messages["ready"], result_mid, public_url))
                    else:
                        result_mid = await send_whatsapp_reply_cloud(
                            to_waid=from_waid,
                            body=f"{vto_messages['ready']} (image hosting unavailable)",
                            phone_number_id=outbound_pnid,
                        )
                        out_msgs.append(("text", f"{vto_messages['ready']} (image hosting unavailable)", result_mid))

                    _set({"active": False})
                    logging.info("[VTO][DONE] session=%s (state cleared)", session_key)

                except Exception:
                    logging.exception("[VTO] Generation failed")
                    fail_mid = await send_whatsapp_reply_cloud(
                        to_waid=from_waid,
                        body=vto_messages["error"],
                        phone_number_id=outbound_pnid,
                    )
                    out_msgs.append(("text", vto_messages["error"], fail_mid))
                    _set({})  # reset state
                    logging.info("[VTO][RESET] session=%s due to failure", session_key)

                logging.info(
                    "[VTO] post-handle (normal flow) | session=%s active=%s step=%s person=%s garment_bytes=%s garment_url=%s",
                    session_key, _get().get("active"), _get().get("step"),
                    "yes" if _get().get("person_image") else "no",
                    "yes" if _get().get("garment_image") else "no",
                    "yes" if _get().get("garment_image_url") else "no",
                )
                return True, out_msgs

        except Exception:
            logging.exception("[VTO] image handling error")

    # No VTO action taken
    logging.info(
        "[VTO] post-handle (normal flow) | session=%s active=%s step=%s person=%s garment_bytes=%s garment_url=%s",
        session_key, vto_state.get("active"), vto_state.get("step"),
        "yes" if vto_state.get("person_image") else "no",
        "yes" if vto_state.get("garment_image") else "no",
        "yes" if vto_state.get("garment_image_url") else "no",
    )
    return False, out_msgs


@router.post("/webhook")
async def receive_cloud_webhook(request: Request):
    entities: dict = {}
    print('Meta webhook..................')
    """Handle inbound messages from Meta Cloud API (value.messages)."""
    raw = await request.body()

    # Parse data first to check payload type
    try:
        data = await request.json()
    except Exception:
        return {"status": "invalid_json"}

    logging.info(f"[CLOUD] Incoming payload: {data}")

    # Check if it's a custom payload format
    is_custom_payload = "whatsapp" in data and "messages" in data.get("whatsapp", {})

    # Only validate signature for official Meta webhooks
    if not is_custom_payload:
        if not _valid_signature(META_APP_SECRET, raw, request.headers.get("X-Hub-Signature-256", "")):
            logging.warning("[CLOUD] Invalid webhook signature")
            return {"status": "forbidden"}
    else:
        logging.info("[CLOUD] Custom payload detected - skipping signature validation")

    msgs_to_process = []
    outbound_pnid = None
    business_number = ""
    contact_name_map: dict[str, str | None] = {}

    # ---- small logging helper for VTO snapshots ----
    def _log_vto_state_snapshot(session_key: str, label: str):
        try:
            st = _get_vto_state(session_key) or {}
        except Exception:
            st = {}
        active = bool(st.get("active"))
        step = st.get("step")
        has_person = st.get("person_image") is not None
        has_garment_bytes = st.get("garment_image") is not None
        has_garment_url = bool(st.get("garment_image_url"))
        logging.info(
            f"[VTO] {label} | session={session_key} "
            f"active={active} step={step} "
            f"person={'yes' if has_person else 'no'} "
            f"garment_bytes={'yes' if has_garment_bytes else 'no'} "
            f"garment_url={'yes' if has_garment_url else 'no'}"
        )

    # Handle standard Meta Cloud API format
    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            meta = value.get("metadata", {})  # display_phone_number, phone_number_id
            msgs = value.get("messages", [])

            if msgs:
                business_number = _normalize_business_number(meta.get("display_phone_number"))
                outbound_pnid = meta.get("phone_number_id")

                logging.info("=%"*100)
                logging.info(f"[CLOUD] Busieness Number : {business_number}")
                logging.info("=%"*100)
                logging.info(f"[CLOUD] Phone Number ID : {outbound_pnid}")

                # If display_phone_number is missing, fetch it via phone_number_id
                if not business_number:
                    pnid = meta.get("phone_number_id")
                    if pnid:
                        business_number = await _lookup_display_phone_number_digits(pnid)

                if not business_number:
                    logging.error("[CLOUD] Could not resolve business number from webhook metadata.")
                    return {"status": "no_business_number"}

                msgs_to_process.extend(msgs)

            for c in value.get("contacts", []):
                wa = (c.get("wa_id") or "").strip()
                nm = (c.get("profile") or {}).get("name")
                if wa:
                    contact_name_map[wa] = nm

    # Handle custom payload format
    if not msgs_to_process:
        whatsapp_data = data.get("whatsapp", {})
        custom_msgs = whatsapp_data.get("messages", [])
        if custom_msgs:
            # ✅ resolve business number
            bn = (
                whatsapp_data.get("business_number")
                or whatsapp_data.get("to")
                or whatsapp_data.get("display_phone_number")
            )
            business_number = _normalize_business_number(bn)
            outbound_pnid = whatsapp_data.get("phone_number_id")

            if not business_number:
                pnid = whatsapp_data.get("phone_number_id")
                if pnid:
                    business_number = await _lookup_display_phone_number_digits(pnid)

            if not business_number:
                logging.error("[CLOUD] Custom payload missing business number")
                return {"status": "no_business_number"}

            logging.info(f"[CLOUD] Using business_number: '{business_number}' for custom payload")

            for custom_msg in custom_msgs:
                content = custom_msg.get("content", {})
                standard_msg = {
                    "id": custom_msg.get("sid"),
                    "from": custom_msg.get("from", ""),
                    "type": content.get("type", "unknown"),
                }
                if standard_msg["type"] == "text":
                    standard_msg["text"] = content.get("text", {})
                msgs_to_process.append(standard_msg)

    if not msgs_to_process:
        return {"status": "no_messages"}

    # Helper: get assistant image_url for a msg_id from transcript
    def _find_assistant_image_url(messages: list, msg_id: str | None) -> str | None:
        if not msg_id:
            return None
        for m in reversed(messages or []):
            if m.get("role") == "assistant" and m.get("msg_id") == msg_id:
                return ((m.get("meta") or {}).get("image_url"))
        return None

    # Process all messages with unified structure
    for msg in msgs_to_process:
        msg_id = msg.get("id") or msg.get("wamid")

        if msg_id in processed_meta_msg_ids:
            continue

        processed_meta_msg_ids.add(msg_id)
        if len(processed_meta_msg_ids) > 5000:
            processed_meta_msg_ids.clear()

        from_waid = (msg.get("from") or "").strip()
        mtype = msg.get("type")
        text_msg = msg.get("text", {}).get("body", "").strip() if mtype == "text" else f"[{mtype} message]"

        # --- HD image handshake guard: ignore the temporary "unsupported" event (code 131051)
        errs = msg.get("errors") or []
        if mtype == "unsupported":
            codes = {e.get("code") for e in errs if isinstance(e, dict)}
            if 131051 in codes:
                logging.info("[CLOUD] Ignoring temporary 'unsupported' (131051) — HD image handshake.")
                continue

        replied_message_id = None
        try:
            if "context" in msg:
                replied_message_id = msg["context"].get("id")
        except Exception:
            replied_message_id = None

        context_obj = msg.get("context")

        # --------------------------- Swipe Reply Flow ---------------------------
        if context_obj and context_obj.get("id"):
            logging.info("=" * 100)
            logging.info("========= Swipe Reply to Bot =========")
            logging.info(f"User replied to message ID: {context_obj.get('id')}")
            logging.info("=" * 100)
            logging.info(f"======== {from_waid}")
            logging.info("=" * 100)

            async for db in get_db():
                try:
                    tenant_id = await get_tenant_id_by_phone(business_number, db)
                    tenant_name = await get_tenant_name_by_phone(business_number, db) or "Your Shop"

                    if not tenant_id:
                        logging.error(f"[CLOUD] No tenant found for business number: '{business_number}'")
                        return {"status": "no_tenant"}

                    # Persist inbound (user swipe)
                    sender_name = contact_name_map.get(from_waid)
                    customer = await get_or_create_customer(
                        db,
                        tenant_id=tenant_id,
                        phone=_normalize_waid_phone(from_waid),
                        whatsapp_id=from_waid,
                        name=sender_name,
                    )
                    chat_session = await get_or_open_active_session(db, customer_id=customer.id)
                    phone_norm = _normalize_waid_phone(from_waid)

                    # Build inbound meta (include image URL if the swipe itself carried one)
                    inbound_meta = {"raw": data, "channel": "cloud_api", "reply_to": context_obj.get("id")}
                    inbound_text = text_msg

                    if mtype == "image" and "image" in msg:
                        media_id = msg["image"].get("id")
                        if media_id:
                            try:
                                media_meta = await get_media_url_and_meta(media_id)
                                inbound_meta["image"] = {
                                    "id": media_id,
                                    "mime_type": media_meta.get("mime_type"),
                                    "sha256": media_meta.get("sha256"),
                                    "url": media_meta.get("url"),
                                }
                                inbound_text = media_meta.get("url") or inbound_text
                            except Exception:
                                logging.exception("[CLOUD][SWIPE] Failed to fetch inbound image url")

                    await append_transcript_message(
                        db, chat_session,
                        role="user", text=inbound_text, msg_id=msg_id, direction="in",
                        meta=inbound_meta,
                    )

                    transcript = await get_transcript_by_phone(phone_norm, db)
                    messages = _normalize_messages(transcript)

                    # Also persist minimal swipe text with reply_to
                    await append_transcript_message(
                        db, chat_session,
                        role="user", text=text_msg, msg_id=msg_id, direction="in",
                        meta={"raw": data, "channel": "cloud_api", "reply_to": context_obj.get("id")},
                    )

                    # 1) Caption to resolve the product
                    caption = find_assistant_image_caption_by_msg_id(messages, context_obj.get("id"))
                    if not caption:
                        await send_whatsapp_reply_cloud(
                            to_waid=from_waid,
                            body="Sorry—couldn't link your reply to a product. Please reply directly to the product image you like.",
                            phone_number_id=outbound_pnid,
                        )
                        await db.commit()
                        return {"status": "ok"}

                    logging.info("=" * 20)
                    logging.info(f"=========Caption used for resolve======== {caption}")

                    # 2) Resolve product from caption
                    resolved = await resolve_product_from_caption_async(caption)
                    resolved_name = resolved.get("name")
                    resolved_product_id = resolved.get("product_id")
                    resolved_variant_id = resolved.get("variant_id")
                    resolved_url = resolved.get("product_url")
                    logging.info(
                        f"✅ Resolved swipe → name='{resolved_name}', product_id={resolved_product_id}, "
                        f"variant_id={resolved_variant_id}, url={resolved_url}"
                    )

                    # Language
                    SUPPORTED_LANGUAGES = ["gu-IN", "hi-IN", "en-IN", "en-US"]
                    current_language = customer.preferred_language or "en-IN"
                    if current_language not in SUPPORTED_LANGUAGES:
                        detected = await detect_language(text_msg, "en-IN")
                        current_language = detected[0] if isinstance(detected, tuple) else detected
                        await update_customer_language(db, customer.id, current_language)

                    # Tenant filters
                    tenant_categories = await get_tenant_category_by_phone(business_number, db)
                    tenant_fabric = await get_tenant_fabric_by_phone(business_number, db)
                    tenant_color = await get_tenant_color_by_phone(business_number, db)
                    tenant_occasion = await get_tenant_occasion_by_phone(business_number, db)
                    tenant_size = await get_tenant_size_by_phone(business_number, db)
                    tenant_type = await get_tenant_type_by_phone(business_number, db)

                    # 3) Attributes seed
                    attrs_db = await get_attrs_for_product_async(resolved_product_id, resolved_variant_id)
                    attrs_text = extract_attrs_from_text(
                        caption,
                        allowed_categories=tenant_categories,
                        allowed_fabrics=tenant_fabric,
                        allowed_occasions=tenant_occasion,
                        allowed_colors=tenant_color,
                    )
                    category = attrs_db.get("category") or attrs_text.get("category")
                    fabric = attrs_db.get("fabric") or attrs_text.get("fabric")
                    occasion = attrs_db.get("occasion") or attrs_text.get("occasion")
                    color = attrs_db.get("color") or attrs_text.get("color")
                    is_rental = attrs_db.get("is_rental")
                    if is_rental is None:
                        is_rental = attrs_text.get("is_rental")

                    seed_entities = {"name": resolved_name}
                    if category: seed_entities["category"] = category
                    if fabric: seed_entities["fabric"] = fabric
                    if occasion: seed_entities["occasion"] = occasion
                    if color: seed_entities["color"] = color
                    if is_rental is not None:
                        seed_entities["is_rental"] = bool(is_rental)

                    logging.info(f"[ATTRS] From DB: {attrs_db} | From text: {attrs_text} | Seed: {seed_entities}")

                    # ---- Detect intent (AI) ----
                    try:
                        intent_type, ai_entities, confidence = await detect_textile_intent_openai(
                            text_msg, current_language,
                            allowed_categories=tenant_categories,
                            allowed_fabric=tenant_fabric,
                            allowed_color=tenant_color,
                            allowed_occasion=tenant_occasion,
                            allowed_size=tenant_size,
                            allowed_type=tenant_type,
                        )
                        entities = ai_entities or {}
                        for k, v in (seed_entities or {}).items():
                            if k == "is_rental":
                                if entities.get(k) is None:
                                    entities[k] = v
                            else:
                                if v and not entities.get(k):
                                    entities[k] = v
                        logging.info(f"[ENTITIES] After coalesce with seed: {entities}")
                    except Exception:
                        logging.exception("[CLOUD] AI pipeline (intent) failed")
                        intent_type, confidence = "other", 0.0

                    # -------------------- VTO START (Swipe Reply) --------------------
                    session_key = f"{tenant_id}:whatsapp:wa:{from_waid}"

                    if intent_type == "virtual_try_on":
                        garment_image_url = _find_assistant_image_url(messages, context_obj.get("id"))

                        _set_vto_state(session_key, {
                            "active": True,
                            "step": "need_person",
                            "person_image": None,
                            "garment_image": None,
                            "garment_image_url": garment_image_url,
                            "seed": entities,
                        })
                        logging.info(f"[VTO][START] via SWIPE | session={session_key} | garment_image_url={garment_image_url}")
                        _log_vto_state_snapshot(session_key, "after start (swipe)")

                        need_person_msg = _get_vto_messages(current_language)["need_person"]
                        mid = await send_whatsapp_reply_cloud(
                            to_waid=from_waid, body=need_person_msg, phone_number_id=outbound_pnid
                        )

                        await append_transcript_message(
                            db, chat_session,
                            role="assistant", text=need_person_msg, msg_id=mid, direction="out",
                            meta={"kind": "text", "channel": "cloud_api"}
                        )
                        await db.commit()
                        return {"status": "ok", "mode": "virtual_try_on_started"}

                    # -------------------- Fallback: regular product flow on swipe --------------------
                    try:
                        raw_reply = await analyze_message(
                            text=text_msg,
                            tenant_id=tenant_id,
                            tenant_name=tenant_name,
                            customer_id=customer.id,
                            language=current_language,
                            intent=intent_type,
                            new_entities=entities,
                            intent_confidence=confidence,
                            mode="chat",
                            session_key=session_key,
                        )
                        reply_text = raw_reply.get("reply_text") if isinstance(raw_reply, dict) else str(raw_reply)
                        if isinstance(raw_reply, dict) and raw_reply.get("intent_type") == "greeting":
                            person = (getattr(customer, "name", None) or sender_name or "").strip()
                            reply_text = f"Hello {person},\nHow can I assist you today?" if person and any(
                                ch.isalpha() for ch in person
                            ) else "Hello! How can I assist you today?"
                        collected_entities = raw_reply.get("collected_entities") if isinstance(raw_reply, dict) else {}
                    except Exception:
                        logging.exception("[CLOUD] AI pipeline failed")
                        raw_reply, reply_text, collected_entities = {}, "Sorry, I'm having trouble. I'll get back to you shortly.", {}

                    followup_text = (raw_reply.get("followup_reply") or "").strip() or None
                    products = (raw_reply.get("pinecone_data") or [])[:5]
                    out_msgs: list[tuple[str, str, str | None]] = []
                    sent_count = 0

                    if products:
                        for prod in products:
                            img = _primary_image_for_product(prod)
                            if not img:
                                continue
                            cap = _product_caption(prod)
                            mid = await send_whatsapp_image_cloud(
                                to_waid=from_waid, image_url=img, caption=cap, phone_number_id=outbound_pnid
                            )
                            if mid:
                                sent_count += 1
                                logging.info(f"[PRODUCT] Sent product message_id={mid} to {from_waid}")
                                out_msgs.append(("image", cap, mid, img))

                        if followup_text:
                            await asyncio.sleep(1.0)
                            mid = await send_whatsapp_reply_cloud(to_waid=from_waid, body=followup_text, phone_number_id=outbound_pnid)
                            out_msgs.append(("text", followup_text, mid))
                    else:
                        if reply_text:
                            mid = await send_whatsapp_reply_cloud(to_waid=from_waid, body=reply_text, phone_number_id=outbound_pnid)
                            out_msgs.append(("text", reply_text, mid))
                        if followup_text:
                            mid = await send_whatsapp_reply_cloud(to_waid=from_waid, body=followup_text, phone_number_id=outbound_pnid)
                            out_msgs.append(("text", followup_text, mid))

                    logging.info(f"============== out Messages :{out_msgs}=================")

                    for item in out_msgs:
                        if isinstance(item, (list, tuple)) and len(item) == 4:
                            kind, txt, mid, img_url = item
                        else:
                            kind, txt, mid = item
                            img_url = None

                        meta = {"kind": kind, "channel": "cloud_api"}
                        if kind == "image" and img_url:
                            meta["image_url"] = img_url
                        if kind == "entities":
                            base = mid or msg_id
                            mid = f"{base}:entities"
                            if isinstance(txt, (dict, list)):
                                meta["entities"] = txt
                                txt = json.dumps(txt, ensure_ascii=False)
                            else:
                                txt = str(txt)

                        await append_transcript_message(
                            db, chat_session,
                            role="assistant", text=txt, msg_id=mid, direction="out", meta=meta
                        )

                    await db.commit()
                    return {
                        "status": "ok",
                        "sent_images": sent_count,
                        "sent_followup": any(m[0] == "text" and m[1] == followup_text for m in out_msgs),
                        "fallback": "none" if products else "text"
                    }

                except Exception:
                    logging.exception("[CLOUD] Webhook DB flow failed; rolling back")
                    await db.rollback()
                    return {"status": "error"}
                finally:
                    break

        # ----------------------------- Normal Flow -----------------------------
        else:
            logging.info("*" * 100)
            logging.info("Normal Reply to Bot")

            async for db in get_db():
                try:
                    tenant_id = await get_tenant_id_by_phone(business_number, db)
                    tenant_name = await get_tenant_name_by_phone(business_number, db) or "Your Shop"

                    if not tenant_id:
                        logging.error(f"[CLOUD] No tenant found for business number: '{business_number}'")
                        return {"status": "no_tenant"}

                    # Persist inbound (normal message)
                    sender_name = contact_name_map.get(from_waid)
                    customer = await get_or_create_customer(
                        db,
                        tenant_id=tenant_id,
                        phone=_normalize_waid_phone(from_waid),
                        whatsapp_id=from_waid,
                        name=sender_name,
                    )

                    logging.info(f"========{from_waid}")

                    # Transcript + last assistant
                    phone_norm = _normalize_waid_phone(from_waid)
                    transcript = await get_transcript_by_phone(phone_norm, db)
                    messages = _normalize_messages(transcript)

                    last_assistant = next((m for m in reversed(messages)
                                           if m.get("role") == "assistant" and m.get("msg_id")), None)

                    derived_reply_id = last_assistant.get("msg_id") if last_assistant else None
                    product_text = find_assistant_text_by_msg_id(messages, derived_reply_id) if derived_reply_id else None
                    product_entities = find_entities_by_msg_id(messages, derived_reply_id) if derived_reply_id else None

                    logging.info("=" * 20)
                    logging.info(f"=========Product_text(derived)======== {product_text}")
                    logging.info(f"=========Product_entities(derived)==== {product_entities}")

                    chat_session = await get_or_open_active_session(db, customer_id=customer.id)

                    inbound_meta = {"raw": data, "channel": "cloud_api", "reply_to": None}
                    inbound_text = text_msg

                    if mtype == "image" and "image" in msg:
                        img_obj = msg.get("image", {}) if isinstance(msg, dict) else {}
                        media_id = img_obj.get("id")
                        link = img_obj.get("link")
                        is_local = request.headers.get("X-LOCAL-TEST") == "1"

                        try:
                            if is_local and link:
                                inbound_meta["image"] = {
                                    "id": media_id or "local-test",
                                    "mime_type": img_obj.get("mime_type") or "image/jpeg",
                                    "sha256": img_obj.get("sha256") or "",
                                    "url": link,
                                }
                                inbound_text = link
                            elif media_id:
                                media_meta = await get_media_url_and_meta(media_id)
                                inbound_meta["image"] = {
                                    "id": media_id,
                                    "mime_type": media_meta.get("mime_type"),
                                    "sha256": media_meta.get("sha256"),
                                    "url": media_meta.get("url"),
                                }
                                inbound_text = media_meta.get("url") or inbound_text
                        except Exception:
                            logging.exception("[CLOUD] Failed to fetch inbound image url (normal flow)")

                    elif mtype == "document" and "document" in msg:
                        mime = (msg["document"].get("mime_type") or "").lower()
                        media_id = msg["document"].get("id")

                        if media_id and mime.startswith("image/"):
                            try:
                                media_meta = await get_media_url_and_meta(media_id)
                                inbound_meta["image"] = {
                                    "id": media_id,
                                    "mime_type": media_meta.get("mime_type"),
                                    "sha256": media_meta.get("sha256"),
                                    "url": media_meta.get("url"),
                                }
                                inbound_text = media_meta.get("url") or inbound_text
                            except Exception:
                                logging.exception("[CLOUD] Failed to fetch inbound doc-image url")

                    await append_transcript_message(
                        db, chat_session,
                        role="user", text=inbound_text,
                        msg_id=msg_id, direction="in",
                        meta=inbound_meta,
                    )

                    # VTO session key + current state
                    session_key = f"{tenant_id}:whatsapp:wa:{from_waid}"
                    _log_vto_state_snapshot(session_key, "pre-handle (normal flow)")

                    # Language (needed for VTO messages)
                    SUPPORTED_LANGUAGES = ["gu-IN", "hi-IN", "en-IN", "en-US"]
                    current_language = customer.preferred_language or "en-IN"
                    if current_language not in SUPPORTED_LANGUAGES:
                        detected = await detect_language(text_msg, "en-IN")
                        current_language = detected[0] if isinstance(detected, tuple) else detected
                        await update_customer_language(db, customer.id, current_language)

                    # -------------------- VTO AUTO-START (Heuristic) --------------------
                    if not _is_vto_flow_active(session_key):
                        last_text = (last_assistant or {}).get("text", "") or ""
                        try:
                            need_person_text = _get_vto_messages(current_language)["need_person"]
                        except Exception:
                            need_person_text = ""
                        if need_person_text and need_person_text[:30] in last_text:
                            _set_vto_state(session_key, {
                                "active": True,
                                "step": "need_person",
                                "person_image": None,
                                "garment_image": None,
                            })
                            logging.info(f"[VTO][START] via HEURISTIC | session={session_key} (last message asked for person photo)")
                            _log_vto_state_snapshot(session_key, "after start (heuristic)")

                    pre_state = _get_vto_state(session_key)
                    vto_handled, vto_out_msgs = await _handle_vto_flow(
                        session_key=session_key,
                        text_msg=text_msg,
                        mtype=mtype,
                        msg=msg,
                        from_waid=from_waid,
                        outbound_pnid=outbound_pnid,
                        current_language=current_language,
                        db=db,
                        chat_session=chat_session,
                        msg_id=msg_id,
                        request=request,   # <-- pass request here
                    )
                    post_state = _get_vto_state(session_key)
                    post_state = _get_vto_state(session_key)

                    _log_vto_state_snapshot(session_key, "post-handle (normal flow)")

                    # Extra log: both images present (either before gen or right after)
                    try:
                        p = (post_state or {}).get("person_image")
                        g_bytes = (post_state or {}).get("garment_image")
                        g_url = (post_state or {}).get("garment_image_url")
                        if p is not None and (g_bytes is not None or g_url):
                            logging.info(f"[VTO][READY] Both images present for session={session_key} (person={'bytes'}; garment={'bytes' if g_bytes else 'url'})")
                    except Exception:
                        pass

                    if vto_handled:
                        logging.info(f"[VTO][HANDLED] session={session_key} | messages={len(vto_out_msgs)}")
                        # Persist VTO messages
                        for item in vto_out_msgs:
                            if isinstance(item, (list, tuple)) and len(item) == 4:
                                kind, txt, mid, img_url = item
                            else:
                                kind, txt, mid = item
                                img_url = None

                            meta = {"kind": kind, "channel": "cloud_api"}
                            if kind == "image" and img_url:
                                meta["image_url"] = img_url

                            await append_transcript_message(
                                db, chat_session,
                                role="assistant",
                                text=txt,
                                msg_id=mid,
                                direction="out",
                                meta=meta,
                            )

                        await db.commit()
                        return {"status": "ok", "mode": "virtual_try_on", "sent_messages": len(vto_out_msgs)}

                    # ------------------------ PRIORITIZED TYPE DISPATCH ------------------------

                    # 1) TEXT
                    if mtype == "text" and "text" in msg:
                        tenant_categories = await get_tenant_category_by_phone(business_number, db)
                        tenant_fabric = await get_tenant_fabric_by_phone(business_number, db)
                        tenant_color = await get_tenant_color_by_phone(business_number, db)
                        tenant_occasion = await get_tenant_occasion_by_phone(business_number, db)
                        tenant_size = await get_tenant_size_by_phone(business_number, db)
                        tenant_type = await get_tenant_type_by_phone(business_number, db)

                        try:
                            intent_type, entities, confidence = await detect_textile_intent_openai(
                                text_msg, current_language,
                                allowed_categories=tenant_categories,
                                allowed_fabric=tenant_fabric,
                                allowed_color=tenant_color,
                                allowed_occasion=tenant_occasion,
                                allowed_size=tenant_size,
                                allowed_type=tenant_type,
                            )

                            if product_entities:
                                entities = _merge_entities(entities, product_entities)

                            logging.info(f"[ENTITIES] merged with product_entities (normal flow): {entities}")

                            raw_reply = await analyze_message(
                                text=text_msg,
                                tenant_id=tenant_id,
                                tenant_name=tenant_name,
                                customer_id=customer.id,
                                language=current_language,
                                intent=intent_type,
                                new_entities=entities,
                                intent_confidence=confidence,
                                mode="chat",
                                session_key=session_key,
                            )

                            reply_text = raw_reply.get("reply_text") if isinstance(raw_reply, dict) else str(raw_reply)

                            if isinstance(raw_reply, dict) and raw_reply.get("intent_type") == "greeting":
                                person = (getattr(customer, "name", None) or sender_name or "").strip()
                                reply_text = f"Hello {person},\nHow can I assist you today?" if person and any(
                                    ch.isalpha() for ch in person
                                ) else "Hello! How can I assist you today?"

                            collected_entities = raw_reply.get("collected_entities") if isinstance(raw_reply, dict) else None

                        except Exception:
                            logging.exception("[CLOUD] AI pipeline failed")
                            reply_text = "Sorry, I'm having trouble. I'll get back to you shortly."
                            collected_entities = None
                            raw_reply = {}

                        raw_obj = raw_reply if isinstance(raw_reply, dict) else {}
                        followup_text = (raw_obj.get("followup_reply") or "").strip() or None
                        products = (raw_obj.get("pinecone_data") or [])[:5]
                        sent_count = 0
                        out_msgs: list[tuple[str, str, str | None]] = []

                        if products:
                            for prod in products:
                                img = _primary_image_for_product(prod)
                                if not img:
                                    continue
                                mid = await send_whatsapp_image_cloud(
                                    to_waid=from_waid, image_url=img, caption=_product_caption(prod), phone_number_id=outbound_pnid
                                )
                                if mid:
                                    sent_count += 1
                                    logging.info(f"[PRODUCT] Sent product message_id={mid} to {from_waid}")
                                    out_msgs.append(("image", _product_caption(prod), mid, img))

                            if followup_text:
                                await asyncio.sleep(1.0)
                                mid = await send_whatsapp_reply_cloud(to_waid=from_waid, body=followup_text, phone_number_id=outbound_pnid)
                                out_msgs.append(("text", followup_text, mid))
                        else:
                            if reply_text:
                                mid = await send_whatsapp_reply_cloud(to_waid=from_waid, body=reply_text, phone_number_id=outbound_pnid)
                                out_msgs.append(("text", reply_text, mid))
                            if followup_text:
                                mid = await send_whatsapp_reply_cloud(to_waid=from_waid, body=followup_text, phone_number_id=outbound_pnid)
                                out_msgs.append(("text", followup_text, mid))

                        logging.info(f"============== out Messages :{out_msgs}=================")

                        for item in out_msgs:
                            if isinstance(item, (list, tuple)) and len(item) == 4:
                                kind, txt, mid, img_url = item
                            else:
                                kind, txt, mid = item
                                img_url = None

                            meta = {"kind": kind, "channel": "cloud_api"}
                            save_id = mid
                            save_text = txt

                            if kind == "image":
                                if img_url:
                                    meta["image_url"] = img_url
                                if isinstance(collected_entities, (dict, list)):
                                    meta["entities"] = collected_entities

                            if kind == "entities":
                                base = mid or msg_id
                                save_id = f"{base}:entities"
                                if isinstance(txt, (dict, list)):
                                    meta["entities"] = txt
                                    save_text = json.dumps(txt, ensure_ascii=False)
                                else:
                                    save_text = str(txt)

                            await append_transcript_message(
                                db, chat_session,
                                role="assistant",
                                text=save_text,
                                msg_id=save_id,
                                direction="out",
                                meta=meta,
                            )

                        await db.commit()
                        return {
                            "status": "ok",
                            "sent_images": sent_count,
                            "sent_followup": any(m[0] == "text" and m[1] == followup_text for m in out_msgs),
                            "fallback": "none" if products else "text",
                        }

                    # 2) IMAGE
                    elif mtype == "image" and "image" in msg:
                        # If we reached here, VTO didn't handle it (inactive or not ready)
                        logging.info(f"[VTO] Not handled; falling back to visual search | session={session_key}")
                        try:
                            media_id = msg["image"].get("id")
                            img_bytes = await download_media_bytes(media_id)

                            matches = visual_search_bytes_sync(img_bytes, tenant_id=tenant_id, top_k=20)
                            matches = group_matches_by_product(matches)[:5]

                            out_msgs: list[tuple[str, str, str | None]] = []
                            sent_count = 0

                            for match in matches:
                                md = (match or {}).get("metadata") or {}
                                img = _primary_image_for_product(md) or md.get("image_url")
                                if not img:
                                    continue

                                cap = _product_caption(md)
                                mid = await send_whatsapp_image_cloud(
                                    to_waid=from_waid,
                                    image_url=img,
                                    caption=cap,
                                    phone_number_id=outbound_pnid,
                                )

                                if mid:
                                    sent_count += 1
                                    logging.info(f"[PRODUCT] Sent product message_id={mid} to {from_waid}")
                                    out_msgs.append(("image", cap, mid, img))

                            if sent_count == 0:
                                body = "Sorry, I couldn't find visually similar items."
                                mid = await send_whatsapp_reply_cloud(to_waid=from_waid, body=body, phone_number_id=outbound_pnid)
                                out_msgs.append(("text", body, mid))
                            else:
                                current_language = (customer.preferred_language or "en-IN")
                                ftxt = _visual_followup_text(current_language)
                                await asyncio.sleep(0.2)
                                mid = await send_whatsapp_reply_cloud(to_waid=from_waid, body=ftxt, phone_number_id=outbound_pnid)
                                out_msgs.append(("text", ftxt, mid))

                            for kind, txt, mid, img_url in _iter_out_msgs_with_meta(out_msgs):
                                meta = {"kind": kind, "channel": "cloud_api"}
                                if img_url:
                                    meta["image_url"] = img_url

                                await append_transcript_message(
                                    db, chat_session,
                                    role="assistant",
                                    text=txt,
                                    msg_id=mid,
                                    direction="out",
                                    meta=meta,
                                )

                            await db.commit()
                            return {"status": "ok", "mode": "visual_search", "sent_images": sent_count}

                        except Exception:
                            logging.exception("[CLOUD] Visual search failed")
                            await db.rollback()
                            mid = await send_whatsapp_reply_cloud(to_waid=from_waid, body="Sorry, I couldn't read that image.", phone_number_id=outbound_pnid)

                            await append_transcript_message(
                                db, chat_session, role="assistant", text="(visual search error)",
                                msg_id=mid, direction="out", meta={"kind": "text", "channel": "cloud_api"}
                            )

                            await db.commit()
                            return {"status": "error", "mode": "visual_search_failed"}

                    # 2.5) VIDEO
                    elif mtype == "video" and "video" in msg:
                        VIDEO_UNAVAILABLE_MSG = (
                            "🎥 Video understanding is under development right now. "
                            "Please send a clear photo or text so I can help."
                        )
                        mid = await send_whatsapp_reply_cloud(to_waid=from_waid, body=VIDEO_UNAVAILABLE_MSG, phone_number_id=outbound_pnid)
                        await append_transcript_message(
                            db, chat_session,
                            role="assistant", text=VIDEO_UNAVAILABLE_MSG,
                            msg_id=mid, direction="out",
                            meta={"kind": "text", "channel": "cloud_api", "feature": "video", "status": "unavailable"}
                        )
                        await db.commit()
                        return {"status": "ok", "mode": "video_unavailable"}

                    # 3) DOCUMENT
                    elif mtype == "document" and "document" in msg:
                        doc = msg["document"]
                        mime = (doc.get("mime_type") or "").lower()
                        media_id = doc.get("id")

                        VIDEO_UNAVAILABLE_MSG = (
                            "🎥 Video understanding is under development right now. "
                            "Please send a clear photo or text so I can help."
                        )

                        if media_id and mime.startswith("video/"):
                            mid = await send_whatsapp_reply_cloud(to_waid=from_waid, body=VIDEO_UNAVAILABLE_MSG, phone_number_id=outbound_pnid)
                            await append_transcript_message(
                                db, chat_session,
                                role="assistant", text=VIDEO_UNAVAILABLE_MSG,
                                msg_id=mid, direction="out",
                                meta={"kind": "text", "channel": "cloud_api", "feature": "video", "status": "unavailable", "doc_mime": mime}
                            )
                            await db.commit()
                            return {"status": "ok", "mode": "doc_video_unavailable"}

                        if media_id and mime.startswith("image/"):
                            try:
                                img_bytes = await download_media_bytes(media_id)
                                matches = visual_search_bytes_sync(img_bytes, tenant_id=tenant_id, top_k=20)
                                matches = group_matches_by_product(matches)[:5]

                                out_msgs: list[tuple[str, str, str | None]] = []
                                sent_count = 0

                                for match in matches:
                                    md = (match or {}).get("metadata") or {}
                                    img = _primary_image_for_product(md) or md.get("image_url")
                                    if not img:
                                        continue

                                    cap = _product_caption(md)
                                    mid = await send_whatsapp_image_cloud(
                                        to_waid=from_waid,
                                        image_url=img,
                                        caption=cap,
                                        phone_number_id=outbound_pnid,
                                    )

                                    if mid:
                                        sent_count += 1
                                        out_msgs.append(("image", cap, mid, img))

                                if sent_count == 0:
                                    body = "Sorry, I couldn't find visually similar items."
                                    mid = await send_whatsapp_reply_cloud(to_waid=from_waid, body=body, phone_number_id=outbound_pnid)
                                    out_msgs.append(("text", body, mid))
                                else:
                                    current_language = (customer.preferred_language or "en-IN")
                                    ftxt = _visual_followup_text(current_language)
                                    await asyncio.sleep(0.2)
                                    mid = await send_whatsapp_reply_cloud(to_waid=from_waid, body=ftxt, phone_number_id=outbound_pnid)
                                    out_msgs.append(("text", ftxt, mid))

                                for kind, txt, mid, img_url in _iter_out_msgs_with_meta(out_msgs):
                                    meta = {"kind": kind, "channel": "cloud_api"}
                                    if img_url:
                                        meta["image_url"] = img_url

                                    await append_transcript_message(
                                        db, chat_session,
                                        role="assistant",
                                        text=txt,
                                        msg_id=mid,
                                        direction="out",
                                        meta=meta,
                                    )

                                await db.commit()
                                return {"status": "ok", "mode": "visual_search_doc_image", "sent_images": sent_count}

                            except Exception:
                                logging.exception("[CLOUD] Visual search (document->image) failed")
                                await db.rollback()
                                mid = await send_whatsapp_reply_cloud(to_waid=from_waid, body="Sorry, I couldn't read that file.", phone_number_id=outbound_pnid)

                                await append_transcript_message(
                                    db, chat_session, role="assistant", text="(document visual search error)",
                                    msg_id=mid, direction="out", meta={"kind": "text", "channel": "cloud_api"}
                                )

                                await db.commit()
                                return {"status": "error", "mode": "doc_visual_search_failed"}

                        # Non-image docs
                        ack_text = "Got your file. I'll take a look and get back to you."
                        mid = await send_whatsapp_reply_cloud(to_waid=from_waid, body=ack_text, phone_number_id=outbound_pnid)

                        await append_transcript_message(
                            db, chat_session,
                            role="assistant", text=ack_text, msg_id=mid, direction="out",
                            meta={"kind": "text", "channel": "cloud_api", "doc_mime": mime}
                        )

                        await db.commit()
                        return {"status": "ok", "mode": "document_ack"}

                    # Fallback
                    else:
                        if mtype == "unsupported" and any((e or {}).get("code") == 131051 for e in (msg.get("errors") or [])):
                            logging.info("[CLOUD] Suppressing fallback for transient 'unsupported' (131051).")
                            await db.commit()
                            return {"status": "ignored_unsupported"}

                        mid = await send_whatsapp_reply_cloud(
                            to_waid=from_waid,
                            body="Sorry, I can only read text, images, or files for now.",
                            phone_number_id=outbound_pnid,
                        )

                        await append_transcript_message(
                            db, chat_session,
                            role="assistant", text="(unsupported message type)", msg_id=mid, direction="out",
                            meta={"kind": "text", "channel": "cloud_api", "mtype": mtype}
                        )

                        await db.commit()
                        return {"status": "ok", "mode": "unsupported"}

                except Exception:
                    logging.exception("[CLOUD] Webhook DB flow failed; rolling back")
                    await db.rollback()
                    return {"status": "error"}
                finally:
                    break

    return {"status": "ok"}
