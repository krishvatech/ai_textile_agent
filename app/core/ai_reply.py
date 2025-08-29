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
from sqlalchemy import text as sql_text  # Import if not already present
from sqlalchemy import select  # ✅
from app.db.session import SessionLocal
from app.core.rental_utils import is_variant_available
from app.core.product_search import pinecone_fetch_records
from app.core.phase_ask_inquiry import format_inquiry_reply,fetch_attribute_values,resolve_categories,fetch_attribute_values,format_inquiry_reply
from app.core.asked_now_detector import detect_requested_attributes_async
from app.core.asked_now_detector import detect_requested_attributes_async
import json
import re
import calendar
import httpx
from app.db.models import Rental, ProductVariant, RentalStatus
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from datetime import date, datetime, timedelta, timezone
from datetime import date as _date  # avoid touching globals
IST = timezone(timedelta(hours=5, minutes=30))

load_dotenv()
api_key = os.getenv("GPT_API_KEY")
gpt_model = os.getenv("GPT_MODEL")
sarvam_api_key = os.getenv("SARVAM_API_KEY")
chat_url = os.getenv("SARVAM_CHAT_URL")
model = os.getenv("SARVAM_LLM_MODEL")
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
    "asking_inquiry", "color_preference", "size_query", "fabric_inquiry"
}

# Quick “DD-DD” and “DD” parsers that assume current month & year (IST)
DAY_RANGE_RE  = re.compile(r'(\b\d{1,2})\s*(?:[-–—]|to|till|until|upto|up to|se|tak)\s*(\d{1,2})\b', re.I)
SINGLE_DAY_RE = re.compile(r'\b(\d{1,2})\b')

MONTH_NAME_RE = re.compile(r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b', re.I)
NUM_MONTH_RE   = re.compile(r'\b\d{1,2}[\/\.]\d{1,2}(?:[\/\.]\d{2,4})?\b')

def _has_month(s: str) -> bool:
    t = str(s or "")
    return bool(MONTH_NAME_RE.search(t) or NUM_MONTH_RE.search(t))

def _quick_day_range(text: str) -> tuple[Optional[int], Optional[int]]:
    if not text: return (None, None)
    m = DAY_RANGE_RE.search(text)
    if not m: return (None, None)
    d1, d2 = int(m.group(1)), int(m.group(2))
    if not (1 <= d1 <= 31 and 1 <= d2 <= 31): return (None, None)
    return d1, d2

def _quick_single_day(text: str) -> Optional[int]:
    if not text: return None
    m = SINGLE_DAY_RE.search(text)
    if not m: return None
    d = int(m.group(1))
    return d if 1 <= d <= 31 else None

def _parse_as_date(val) -> date | None:
    """Parse a value into a date. Tries ISO first; then day-first for India."""
    if not val:
        return None
    if isinstance(val, date):
        return val
    s = str(val).strip()

    # Strict ISO first (YYYY-MM-DD)
    try:
        return date.fromisoformat(s)
    except Exception:
        pass

    # Fallback: natural parse with day-first
    if dateparser:
        try:
            # default gives missing pieces (like year) a sensible base
            dt = dateparser.parse(s, dayfirst=True, yearfirst=False, default=datetime.now(IST))
            return dt.date() if dt else None
        except Exception:
            return None

    return None

def _past_date_msg(s: date, e: Optional[date], language: str) -> str:
    lr = (language or 'en-IN').split('-')[0].lower()
    if e:
        if lr == 'hi':
            return f"यह तिथियाँ अतीत में हैं ({s.strftime('%d %b %Y')} से {e.strftime('%d %b %Y')}). कृपया भविष्य की तारीखें भेजें."
        if lr == 'gu':
            return f"આ તારીખો ભૂતકાળની છે ({s.strftime('%d %b %Y')} થી {e.strftime('%d %b %Y')}). કૃપા કરીને ભવિષ્યની તારીખો આપો."
        return f"That date range is in the past ({s.strftime('%d %b %Y')} to {e.strftime('%d %b %Y')}). Please share a future range."
    else:
        if lr == 'hi':
            return f"यह तारीख अतीत में है ({s.strftime('%d %b %Y')}). कृपया भविष्य की तारीख भेजें."
        if lr == 'gu':
            return f"આ તારીખ ભૂતકાળમાં છે ({s.strftime('%d %b %Y')}). કૃપા કરીને ભવિષ્યની તારીખ આપો."
        return f"That date is in the past ({s.strftime('%d %b %Y')}). Please share a future date."

async def _create_rental_if_needed(acc_entities: Dict[str, Any]) -> Optional[int]:
    """
    Creates a Rental row if we have everything we need and haven't created one already.
    Returns rental_id (int) or None.
    """
    # prevent duplicates on repeated "yes"
    if acc_entities.get("rental_id"):
        return acc_entities["rental_id"]

    # must be a rental confirmation with concrete variant + dates
    if not acc_entities.get("is_rental"):
        return None
    vid = acc_entities.get("product_variant_id")
    s   = acc_entities.get("start_date")
    e   = acc_entities.get("end_date")
    if not (vid and s and e):
        return None

    # parse to date -> datetime (00:00)
    s_date = _parse_as_date(s)
    e_date = _parse_as_date(e)
    if not (s_date and e_date):
        return None
    s_dt = datetime.combine(s_date, datetime.min.time())
    e_dt = datetime.combine(e_date, datetime.min.time())

    # work out rental_price: prefer entity, else variant.rental_price
    price = acc_entities.get("rental_price")
    async with SessionLocal() as db:
        if price in (None, "", [], {}):
            try:
                pv = await db.get(ProductVariant, int(vid))
                if pv and pv.rental_price is not None:
                    price = float(pv.rental_price)
            except Exception:
                price = None
        if price is None:
            price = 0.0  # last-resort fallback

        rental = Rental(
            product_variant_id=int(vid),
            rental_start_date=s_dt,
            rental_end_date=e_dt,
            rental_price=float(price),
            status=RentalStatus.active,
        )
        db.add(rental)
        await db.commit()
        await db.refresh(rental)

        acc_entities["rental_id"] = rental.id
        return rental.id
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


# --- REPLACE THIS WHOLE FUNCTION ---
async def FollowUP_Question(
    intent_type: str,
    entities: Dict[str, Any],
    language: Optional[str] = "en-IN",
    session_history: Optional[List[Dict[str, str]]] = None,
    only_fields: Optional[List[str]] = None,
    max_fields: int = 2
) -> str:
    """
    Generates a short, merged follow-up question asking ONLY for specific missing fields,
    powered by Sarvam LLM, with strict Gujarati support & localized fallback.
    """

    # ---- helpers ----
    def _is_missing(val):
        return (val is None) or (val == "") or (isinstance(val, (list, dict)) and not val)

    def _lang_root(lang: Optional[str]) -> str:
        return (lang or "en-IN").split("-")[0].lower()

    # Localized field labels (expand as needed)
    FIELD_LABELS = {
        "en": {
            "is_rental": "rent or buy",
            "occasion": "occasion",
            "fabric": "fabric",
            "size": "size",
            "color": "color",
            "category": "category",
            "quantity": "quantity",
            "start_date": "start date",
            "end_date": "end date",
            "confirmation": "confirmation",
            "price": "price",
            "rental_price": "rental price",
        },
        "gu": {
            "is_rental": "ભાડે કે ખરીદી",
            "occasion": "પ્રસંગ",
            "fabric": "કાપડ",
            "size": "સાઇઝ",
            "color": "રંગ",
            "category": "વર્ગ",
            "quantity": "quantity",
            "start_date": "શરૂઆતની તારીખ",
            "end_date": "અંતીમ તારીખ",
            "confirmation": "કન્ફોર્મ",
            "price": "ભાવ",
            "rental_price": "ભાડું",
        },
        "hi": {
            "is_rental": "किराए या खरीद",
            "occasion": "मौका",
            "fabric": "कपड़ा",
            "size": "साइज़",
            "color": "रंग",
            "category": "श्रेणी",
            "quantity": "संख्या",
            "start_date": "प्रारंभ तिथि",
            "end_date": "समाप्ति तिथि",
            "confirmation": "पुष्टिकरण",
            "price": "कीमत",
            "rental_price": "किराया",
        }
    }

    def _label(field: str, lang: str) -> str:
        labels = FIELD_LABELS.get(lang, FIELD_LABELS["en"])
        return labels.get(field, field)

    async def _sarvam_chat(messages: List[Dict[str, str]],
                           prompt_str: str,
                           model: Optional[str] = None,
                           temperature: float = 0.3,
                           max_tokens: int = 120) -> Optional[str]:
        if not api_key:
            return None

        headers = {
            "Authorization": f"Bearer {sarvam_api_key}",
            "Content-Type": "application/json",
        }

        # 1) Chat-style attempt
        try:
            payload_chat = {
                "model": model, 
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(chat_url, headers=headers, json=payload_chat)
                if r.status_code == 200:
                    data = r.json()
                    content = (data.get("choices", [{}])[0]
                                  .get("message", {})
                                  .get("content"))
                    if content:
                        return content.strip()
        except Exception:
            pass

    # ---- main logic (entity selection) ----
    is_rental_val = entities.get("is_rental", None)

    base_keys = [
        "is_rental", "occasion", "fabric", "size", "color", "category",
        "quantity",
    ] + (["start_date", "end_date"] if is_rental_val is True else []) + ["confirmation"]

    price_keys = (
        ["rental_price"] if is_rental_val is True
        else (["price"] if is_rental_val is False else ["price", "rental_price"])
    )

    entity_priority = base_keys + price_keys
    missing_fields = [k for k in entity_priority if _is_missing(entities.get(k))]

    if only_fields:
        wanted = set(only_fields)
        missing_fields = [k for k in missing_fields if k in wanted]

    if not missing_fields:
        return "બધા વિગત મળી ગયા. આભાર!" if _lang_root(language) == "gu" else \
               "Thank you. I have all the information I need for your request!"

    # Stable order + cap
    entity_priority_sorted = [
        "is_rental", "occasion", "fabric", "size", "color", "category",
        "quantity", "start_date", "end_date", "confirmation", "price", "rental_price"
    ]
    missing_sorted = sorted(
        missing_fields,
        key=lambda x: entity_priority_sorted.index(x) if x in entity_priority_sorted else 999
    )
    missing_short = missing_sorted[: max(1, int(max_fields))]

    # Language prep
    lang = _lang_root(language)
    non_empty = {k: v for k, v in (entities or {}).items() if v not in (None, "", [], {})}
    textile_prompt = (globals().get("Textile_Prompt", "") or "").strip()

    # Localized list of missing fields for the prompt
    missing_human = ", ".join([_label(f, lang) for f in missing_short])

    # Recent context
    session_text = ""
    if session_history:
        relevant = session_history[-5:]
        conv_lines = []
        for m in relevant:
            role = (m.get("role") or "").capitalize() or "User"
            content = m.get("content") or ""
            conv_lines.append(f"{role}: {content}")
        session_text = "Conversation so far:\n" + "\n".join(conv_lines) + "\n"

    # ---- Build language-strong prompts ----
    if lang == "gu":
        system_msg = (
            "તમે એક ટેક્સટાઇલ દુકાનના મદદગાર છો. હંમેશાં ખૂબ સંક્ષિપ્ત, નમ્ર અને "
            "ગુજરાતી લિપિમાં જ જવાબ આપો. હિંગ્લિશ/અંગ્રેજી શબ્દો ટાળો."
        )
        user_instr = (
            f"{textile_prompt}\n"
            f"{session_text}"
            f"હમણાં સુધી મળેલી માહિતી (કી=ઇંગ્લિશ, મૂલ્ય=વેલ્યૂ): {json.dumps(non_empty, ensure_ascii=False)}\n"
            f"હવે ફક્ત આ બાબતો પૂછો: {missing_human}.\n"
            "એક જ સવાલ, ખૂબ જ ટૂંકો અને સહજ. અન્ય કઈ માહિતી ન પૂછશો."
        )
    elif lang == "hi":
        system_msg = (
            "आप एक टेक्सटाइल दुकान के सहायक हैं। हमेशा संक्षिप्त, विनम्र और "
            "देवनागरी हिन्दी में ही उत्तर दें। हिंग्लिश/अंग्रेज़ी शब्द न प्रयोग करें।"
        )
        user_instr = (
            f"{textile_prompt}\n"
            f"{session_text}"
            f"अब तक मिली जानकारी (keys in English): {json.dumps(non_empty, ensure_ascii=False)}\n"
            f"अब केवल यह पूछें: {missing_human}.\n"
            "एक ही सवाल, बहुत छोटा और स्वाभाविक। और कोई बात न पूछें।"
        )
    else:
        system_msg = (
            "You are a textile shop assistant. Be concise, polite, and reply ONLY in English."
        )
        user_instr = (
            f"{textile_prompt}\n"
            f"{session_text}"
            f"Details so far: {json.dumps(non_empty, ensure_ascii=False)}\n"
            f"Ask ONLY for: {missing_human}.\n"
            "Return a single, very short question. Do not ask anything else."
        )

    # Compose both chat messages and a plain prompt (for fallback endpoint)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_instr},
    ]
    plain_prompt = system_msg + "\n\n" + user_instr

    # ---- Call Sarvam ----
    llm_text = await _sarvam_chat(messages, plain_prompt)

    # ---- Localized fallback (if Sarvam fails) ----
    if not llm_text:
        if len(missing_short) == 1:
            f1 = _label(missing_short[0], lang)
            if lang == "gu":
                return f"કૃપા કરીને તમારી {f1} જણાવશો?"
            elif lang == "hi":
                return f"कृपया अपनी {f1} बताइए?"
            else:
                return f"Could you please share your {f1}?"
        else:
            if lang == "gu":
                fields = " અને ".join([_label(x, lang) for x in [*missing_short[:-1], missing_short[-1]]]) \
                         if len(missing_short) == 2 else \
                         ", ".join([_label(x, lang) for x in missing_short[:-1]]) + f" અને {_label(missing_short[-1], lang)}"
                return f"કૃપા કરીને તમારી {fields} જણાવશો?"
            elif lang == "hi":
                fields = " और ".join([_label(x, lang) for x in [*missing_short[:-1], missing_short[-1]]]) \
                         if len(missing_short) == 2 else \
                         ", ".join([_label(x, lang) for x in missing_short[:-1]]) + f" और {_label(missing_short[-1], lang)}"
                return f"कृपया अपनी {fields} बताइए?"
            else:
                fields = " and ".join([_label(x, lang) for x in [*missing_short[:-1], missing_short[-1]]]) \
                         if len(missing_short) == 2 else \
                         ", ".join([_label(x, lang) for x in missing_short[:-1]]) + f" and {_label(missing_short[-1], lang)}"
                return f"Could you please share your {fields}?"

    return llm_text.strip()



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
    # keep spaces for fabric so "jimmy chu" stays "jimmy chu"
    preserve_space_keys = ["category", "size", "occasion", "fabric", "color"]
    for k, v in (entities or {}).items():
        if isinstance(v, str):
            s = v.strip().lower()
            new_entities[k] = s if k in preserve_space_keys else s.replace(" ", "")
        else:
            new_entities[k] = v
    return new_entities

# --- helper: collapse variants so every product appears once ---
def dedupe_products(pinecone_data, max_items: int = 5):
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
    for g in grouped.values():  # preserves first-seen order
        base = g["base"]
        base["available_colors"] = sorted(c for c in g["colors"] if c)
        base["available_sizes"]  = sorted(s for s in g["sizes"] if s)
        base["image_urls"] = list(g["images"])
        if g["min_price"] is not None: base["price"] = g["min_price"]
        if g["min_rent"]  is not None: base["rental_price"] = g["min_rent"]
        out.append(base)

    # ✅ hard-cap to 5 (default)
    return out[: max_items]


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
            "product_name","quantity","type","price","rental_price",
            "user_name","confirmation"
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
    language: str,
) -> str:
    
    # Text-based type detection if NER missed it
    text_lower = (text or "").lower()
    detected_type = None
    if any(word in text_lower for word in ["for women", "women's", "female", "ladies"]):
        detected_type = "women"
    elif any(word in text_lower for word in ["for men", "men's", "male","men"]):
        detected_type = "men"
    elif any(word in text_lower for word in ["for kids", "children", "child"]):
        detected_type = "kids"
    elif any(word in text_lower for word in ["for boys", "boy's"]):
        detected_type = "boys"
    elif any(word in text_lower for word in ["for girls", "girl's"]):
        detected_type = "girls"
    
    # Check if this is a type-based category request
    requested_type = detected_type or acc_entities.get("type")
    asking_categories = requested_type and not any(acc_entities.get(k) for k in ["category", "color", "fabric", "size"])
    
    if requested_type and asking_categories:
        try:
            categories = await get_categories_by_type(db, tenant_id, requested_type)
            
            if categories:
                bullets = "\n".join(f"• {c}" for c in categories[:12])
                lr = (language or "en-IN").split("-")[0].lower()
                
                if lr == "hi":
                    return f"{requested_type.title()} के लिए उपलब्ध कैटेगरी:\n{bullets}\nकिस कैटेगरी में देखना चाहेंगे?"
                elif lr == "gu":
                    return f"{requested_type.title()} માટે ઉપલબ્ધ કેટેગરી:\n{bullets}\nકઈ કેટેગરીમાં જોવું છે?"
                else:
                    return f"Available categories for {requested_type}:\n{bullets}\nWhich category would you like to explore?"
            else:
                lr = (language or "en-IN").split("-")[0].lower()
                if lr == "hi":
                    return f"खराब, {requested_type} के लिए फिलहाल कोई प्रोडक्ट उपलब्ध नहीं है."
                elif lr == "gu":
                    return f"માફ કરશો, {requested_type} માટે હાલમાં કોઈ પ્રોડક્ટ ઉપલબ્ધ નથી."
                else:
                    return f"Sorry, no products available for {requested_type} currently."
                    
        except Exception as e:
            logging.error(f"Error in type-based categories: {e}")
    
    # Continue with existing logic for regular inquiries...
    try:
        asked_now = await detect_requested_attributes_async(text or "", acc_entities or {})
    except Exception:
        asked_now = []

    tl = (text or "").lower()
    # These run EVEN IF the detector returned "category", so clear user intent wins.
    if re.search(r"(?:\b|^)(kapad|kapda|kapde|kapdu)\b", tl) or ("કાપડ" in tl) or ("ફેબ્રિક" in tl) or re.search(r"\bfabrics?\b", tl):
        asked_now = ["fabric"]
    elif re.search(r"\b(colors?|colours?)\b", tl) or ("રંગ" in tl) or ("rang" in tl):
        asked_now = ["color"]
    elif re.search(r"\bsize(s)?\b", tl) or ("સાઇઝ" in tl) or ("માપ" in tl):
        asked_now = ["size"]
    elif re.search(r"\b(price|rate|cost|mrp)\b", tl) or ("કિંમત" in tl) or ("ભાવ" in tl):
        asked_now = ["price"]

    # Final fallback only if still nothing inferred
    if not asked_now:
        asked_now = ["category"]

    print("asked_now  =", asked_now)


    needs_category = any(k in asked_now for k in ("price", "rental_price")) and not (acc_entities or {}).get("category")

    if needs_category:
        try:
            cats = await resolve_categories(db, tenant_id, {})
        except Exception:
            cats = []

        if cats:
            bullets = "\n".join(f"• {c}" for c in cats[:12])
            lr = (language or "en-IN").split("-")[0].lower()
            if lr == "hi":
                return f"कृपया दाम बताने के लिए एक कैटेगरी चुनें:\n{bullets}"
            elif lr == "gu":
                return f"કિંમત જણાવવા માટે કૃપા કરીને એક કેટેગરી પસંદ કરો:\n{bullets}"
            else:
                return f"Please choose a category for the price range:\n{bullets}"

        lr = (language or "en-IN").split("-")[0].lower()
        if lr == "hi":
            return "कृपया दाम बताने के लिए कैटेगरी बताइए."
        elif lr == "gu":
            return "કિંમત જણાવવા માટે કૃપા કરીને કેટેગરી જણાવો."
        return "Please tell me the category for the price range."

    values = await fetch_attribute_values(db, tenant_id, asked_now, acc_entities or {})
    return format_inquiry_reply(values, acc_entities, language=language)


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

DEPT_SET = {"women","men","girls","boys","kids"}

async def db_type_for_category(db: AsyncSession, tenant_id: int, category: str) -> str | None:
    """Return dominant department (type) for a category from products table."""
    if not category:
        return None
    result = await db.execute(
        text("""
            SELECT type
            FROM products
            WHERE tenant_id = :t
              AND lower(category) = lower(:c)
              AND type IS NOT NULL
            GROUP BY type
            ORDER BY COUNT(*) DESC
            LIMIT 1
        """),
        {"t": tenant_id, "c": category}
    )
    row = result.fetchone()
    if not row:
        return None
    t = (row[0] or "").strip().lower()
    return t if t in DEPT_SET else None

async def get_categories_by_type(db: AsyncSession, tenant_id: int, type_filter: str) -> List[str]:
    """Get distinct categories for a specific type (men/women/kids/etc.)"""
    try:
        result = await db.execute(
            text("""
                SELECT DISTINCT category 
                FROM products 
                WHERE tenant_id = :tid 
                  AND LOWER(type) = LOWER(:type_filter)
                  AND category IS NOT NULL 
                  AND category != ''
                ORDER BY category
            """),
            {"tid": tenant_id, "type_filter": type_filter}
        )
        
        categories = [row[0] for row in result.fetchall() if row[0]]
        return categories
        
    except Exception as e:
        logging.error(f"Error in get_categories_by_type: {e}")
        return []


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
    prev_entities = dict(acc_entities)  # <--- ADD THIS LINE
    # --- Clean and merge new entities into memory (critical!) ---
    raw_new_entities = new_entities or {}
    if not raw_new_entities.get("start_date") and not raw_new_entities.get("end_date"):
        d1, d2 = _quick_day_range(text or "")
        if d1 and d2:
            today = datetime.now(IST).date()
            y, m = today.year, today.month
            last_day = calendar.monthrange(y, m)[1]
            d1 = min(d1, last_day)
            d2 = min(d2, last_day)
            s = date(y, m, d1)
            e = date(y, m, d2)
            if e < s:
                # if user typed 28-2 etc., push end into next month sensibly
                nm, ny = (m + 1, y) if m < 12 else (1, y + 1)
                last_next = calendar.monthrange(ny, nm)[1]
                e = date(ny, nm, min(d2, last_next))
            raw_new_entities["start_date"] = s.isoformat()
            raw_new_entities["end_date"]   = e.isoformat()
            # if classifier said "other", gently push into availability flow
            intent_type = intent_type or "availability_check"

    # Optional: plain “25” as follow-up → treat as end date if start exists, else start date
    expecting_end = bool((prev_entities or {}).get("start_date") and not (prev_entities or {}).get("end_date"))
    if expecting_end and not raw_new_entities.get("start_date") and not raw_new_entities.get("end_date"):
        only_day = _quick_single_day(text or "")
        if only_day:
            today = datetime.now(IST).date()
            y, m = today.year, today.month
            last_day = calendar.monthrange(y, m)[1]
            only_day = min(only_day, last_day)
            raw_new_entities["end_date"] = date(y, m, only_day).isoformat()
            intent_type = intent_type or "availability_check"
    # --- Force sensible year if the USER didn't type a year (avoid stale 2024 from NER)
    msg_has_year = bool(re.search(r"\b(19|20)\d{2}\b", text or ""))

    if not msg_has_year:
        today = datetime.now(IST).date()
        for k in ("start_date", "end_date"):
            v = raw_new_entities.get(k)
            if not v:
                continue
            d = _parse_as_date(v)
            if not d:
                continue
            if not _has_month(text or ""):
                # Month not typed → keep current month, current year; cap the day safely
                y, m = today.year, today.month
                last = calendar.monthrange(y, m)[1]
                d = date(y, m, min(d.day, last))
            else:
                # Month typed → snap only the year; if still past, bump to next year
                candidate = d.replace(year=today.year)
                if candidate < today:
                    candidate = candidate.replace(year=today.year + 1)
                d = candidate
            raw_new_entities[k] = d.isoformat()
    clean_new_entities = filter_non_empty_entities(raw_new_entities)
    cat_now = (clean_new_entities.get("category") or acc_entities.get("category"))
    if cat_now and not clean_new_entities.get("type"):
        async with SessionLocal() as db:
            t = await db_type_for_category(db, tenant_id, cat_now)
        if t:
            clean_new_entities["type"] = t
    new_cat = (new_entities or {}).get("category")
    if new_cat:
        prev_cat = acc_entities.get("category")
        if not prev_cat or _lc(prev_cat) != _lc(new_cat):
            KEEP_ON_CATEGORY_CHANGE = {"category", "type", "is_rental"}
            for dep in list(acc_entities.keys()):
                if dep not in KEEP_ON_CATEGORY_CHANGE:
                    acc_entities.pop(dep, None)
            acc_entities["category"] = new_cat
    if intent_type == "availability_check":
        prev_start = (prev_entities or {}).get("start_date")
        prev_end = (prev_entities or {}).get("end_date")
        
        # If we had a start_date but no end_date, and NER extracted same date for both
        if (prev_start and not prev_end and 
            clean_new_entities.get("start_date") == clean_new_entities.get("end_date") and
            clean_new_entities.get("start_date")):
            # Remove start_date from new entities to prevent overwriting
            clean_new_entities.pop("start_date", None)
            logging.info(f"DEBUG: Prevented start_date overwrite. Keeping: {prev_start}")

    acc_entities = merge_entities(acc_entities, clean_new_entities)
    # acc_entities = merge_entities(acc_entities, clean_new_entities)
    

    logging.info(f"Collected entities AFTER MERGE: {acc_entities}")
    
    # Reset dependent filters if category changed
    
            
    def _commit():
        session_memory[sk] = history
        session_entities[sk] = acc_entities
            
            
    logging.info(f"intent_type(detected)..... {detected_intent}")
    logging.info(f"intent_type(resolved)..... {intent_type}")
    try:
        if intent_type == "asking_inquiry":
            # do we already have a category in memory?
            has_category_ctx = bool(acc_entities.get("category"))

            # did user explicitly flip rent/buy this turn?
            explicit_rent_buy = (clean_new_entities.get("is_rental") is not None)

            # any other concrete attribute provided this turn?
            concrete_attr_now = any(
                (clean_new_entities.get(k) not in (None, "", [], {}))
                for k in ("color", "fabric", "size", "occasion", "price", "rental_price", "quantity", "type")
            )

            # option-list / WH question? (EN + HI + GU cues)
            tl = (text or "").lower()
            is_option_list_q = ("?" in tl) or any(w in tl for w in [
                "which", "what", "konsa", "kaunse", "kounsa",
                "कौन", "कौनसा", "कौनसे", "क्या",
                "કયા", "કયું", "કઈ", "શું",
            ])

            if has_category_ctx and (explicit_rent_buy or concrete_attr_now) and not is_option_list_q:
                intent_type = "product_search"
                logging.info("Override: context category + refinement → product_search")
    except Exception as e:
        logging.exception("Refinement override failed: %s", e)
    in_rental_context = (
        (acc_entities.get("is_rental") is True)
        or bool(acc_entities.get("product_variant_id"))
        or last_main_intent == "availability_check"
    )
    if intent_type in {"stock_check", "price_check"} and in_rental_context:
        intent_type = "availability_check"
    # --- greeting
    if intent_type == "greeting":
        # 🧹 HARD RESET: wipe everything we've collected for this session
        session_entities[sk] = {}
        last_main_intent_by_session.pop(sk, None)
        acc_entities = {}  # keep in-sync with memory
        nm = None
        try:
            nm = (new_entities or {}).get("user_name") or (new_entities or {}).get("name")
        except Exception:
            nm = None
        nm = (nm or "").strip()
        reply = f"Hello {nm},\nHow can I assist you today?" if nm else "Hello! How can I assist you today?"
        history.append({"role": "assistant", "content": reply})
         # Persist the cleared state
        session_memory[sk] = history         # (ok to keep history)
        session_entities[sk] = acc_entities  # {} after reset
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
        # -------------------------------
        # Build filters from this turn + memory
        # -------------------------------
        turn_filters = {
            k: v for k, v in (locals().get("clean_new_entities") or {}).items()
            if v not in (None, "", [], {}) and k in ("category","color","fabric","size","is_rental","occasion","type")
        }
        # Fallback from memory for common facets if missing this turn
        for k in ("category", "is_rental", "color", "fabric", "size", "occasion","type"):
            if k not in turn_filters and (acc_entities.get(k) not in (None, "", [], {})):
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
        # Normalize → Pinecone search
        filtered_entities       = filter_non_empty_entities(turn_filters)
        filtered_entities_norm  = normalize_entities(filtered_entities)
        filtered_entities_norm  = clean_entities_for_pinecone(filtered_entities_norm)
        pinecone_data = await pinecone_fetch_records(filtered_entities_norm, tenant_id)
        _raw_matches = list(pinecone_data or [])
        # ---------------------------------------
        # Resolve a single concrete variant (if any)
        # ---------------------------------------
        def _match_attr(m, key):
            return (str(m.get(key) or "").strip().lower()
                    == str(turn_filters.get(key) or "").strip().lower()) if turn_filters.get(key) else True
        cands = [m for m in _raw_matches if _match_attr(m, "color") and _match_attr(m, "size") and _match_attr(m, "fabric")]
        resolved_variant_id = None
        if len(cands) == 1 and cands[0].get("variant_id"):
            resolved_variant_id = cands[0]["variant_id"]
        elif len(_raw_matches) == 1 and _raw_matches[0].get("variant_id"):
            resolved_variant_id = _raw_matches[0]["variant_id"]
        if resolved_variant_id:
            acc_entities["product_variant_id"] = resolved_variant_id
        pinecone_data = dedupe_products(pinecone_data)
        requested_type = (clean_new_entities.get("type") or acc_entities.get("type"))
        asking_categories = not any(turn_filters.get(k) for k in ["category", "color", "fabric", "size"]) and requested_type

        if requested_type and asking_categories:
            # User is asking "what products do you have for men?" - show categories for that type
            try:
                async with SessionLocal() as db:
                    categories = await get_categories_by_type(db, tenant_id, requested_type)
                
                if categories:
                    lr = (language or "en-IN").split("-")[0].lower()
                    bullets = "\n".join(f"• {c}" for c in categories[:12])
                    
                    if lr == "hi":
                        reply_text = f"{requested_type.title()} के लिए उपलब्ध कैटेगरी:\n{bullets}\nकिस कैटेगरी में देखना चाहेंगे?"
                    elif lr == "gu":
                        reply_text = f"{requested_type.title()} માટે ઉપલબ્ધ કેટેગરી:\n{bullets}\nકઈ કેટેગરીમાં જોવું છે?"
                    else:
                        reply_text = f"Available categories for {requested_type}:\n{bullets}\nWhich category would you like to explore?"
                    
                    history.append({"role": "assistant", "content": reply_text})
                    _commit()
                    
                    return {
                        "intent_type": intent_type,
                        "language": language,
                        "tenant_id": tenant_id,
                        "history": history,
                        "collected_entities": acc_entities,
                        "reply_text": reply_text,
                        "categories": categories
                    }
                else:
                    lr = (language or "en-IN").split("-")[0].lower()
                    if lr == "hi":
                        reply_text = f"खुशी, {requested_type} के लिए फिलहाल कोई प्रोडक्ट उपलब्ध नहीं है."
                    elif lr == "gu":
                        reply_text = f"માફ કરશો, {requested_type} માટે હાલમાં કોઈ પ્રોડક્ટ ઉપલબ્ધ નથી."
                    else:
                        reply_text = f"Sorry, no products available for {requested_type} currently."
                    
                    history.append({"role": "assistant", "content": reply_text})
                    _commit()
                    
                    return {
                        "intent_type": intent_type,
                        "language": language,
                        "tenant_id": tenant_id,
                        "history": history,
                        "collected_entities": acc_entities,
                        "reply_text": reply_text
                    }
                    
            except Exception as e:
                logging.error(f"Error fetching categories by type: {e}")
        
        # =========================
        # ZERO-RESULTS SMART FALLBACK (color + fabric + size + occasion)
        # =========================
        if not pinecone_data:
            lr = (language or "en-IN").split("-")[0].lower()
            # Polite base message
            if lr.startswith("hi"):
                reply_text = "क्षमा करें, इस खोज के लिए अभी कोई विकल्प नहीं मिला।"
            elif lr.startswith("gu"):
                reply_text = "માફ કરશો, આ શોધ માટે હાલમાં કોઈ વિકલ્પ મળ્યો નથી."
            else:
                reply_text = "Sorry, I couldn’t find options for that combination."
            suggested_parts = []
            ask_focus = None   # which attribute we'll ask about (only one follow-up)
            req_cat = (acc_entities or {}).get("category")
            # Priority for which follow-up to ask (color first as requested)
            ATTR_PRIORITY = ["color", "fabric", "size", "occasion"]
            # Add a short "not available" clarifier for the chosen attribute
            def _append_not_available(attr: str, asked_val: str):
                nonlocal reply_text
                if not (req_cat and asked_val):
                    return
                if lr.startswith("hi"):
                    reply_text += f" {req_cat} में '{asked_val}' नहीं दिख रहा."
                elif lr.startswith("gu"):
                    reply_text += f" {req_cat} માં '{asked_val}' ઉપલબ્ધ નથી."
                else:
                    reply_text += f" '{asked_val}' isn’t available for {req_cat}."
            try:
                async with SessionLocal() as db:
                    for attr in ATTR_PRIORITY:
                        # Only suggest alternatives if user explicitly constrained this attribute
                        asked_val = (acc_entities or {}).get(attr)
                        if asked_val in (None, "", [], {}):
                            continue
                        # Remove this attribute from filters to discover what DOES exist for it
                        ents_no_attr = dict(acc_entities or {})
                        ents_no_attr.pop(attr, None)
                        values = await fetch_attribute_values(db, tenant_id, [attr], ents_no_attr)
                        options = (values or {}).get(attr) or []
                        # Drop the asked value itself and blanks; cap to keep message short
                        asked_low = str(asked_val).strip().lower()
                        options = [o for o in options if str(o).strip() and str(o).strip().lower() != asked_low][:8]
                        if options:
                            # Localized line like:
                            #  - "Available colors include …"
                            #  - "Available fabrics include …"
                            #  - "Available sizes include …"
                            #  - "Available occasions include …"
                            part = format_inquiry_reply({attr: options}, ctx=acc_entities, language=language)
                            suggested_parts.append(part)
                            # Choose the first attribute (by priority) as the one to ask about
                            if ask_focus is None:
                                ask_focus = attr
                                _append_not_available(attr, asked_val)
            except Exception:
                suggested_parts = []
                ask_focus = None
            if suggested_parts:
                # Localized single follow-up based on ask_focus
                if lr.startswith("hi"):
                    ASK_LINES = {
                        "color":    "कौन-सा रंग दिखाऊँ?",
                        "fabric":   "कौन-सा फ़ैब्रिक दिखाऊँ?",
                        "size":     "कौन-सा साइज दिखाऊँ?",
                        "occasion": "किस अवसर के लिए दिखाऊँ?",
                    }
                elif lr.startswith("gu"):
                    ASK_LINES = {
                        "color":    "કયો રંગ બતાવું?",
                        "fabric":   "કયું ફેબ્રિક બતાવું?",
                        "size":     "કયો સાઇઝ બતાવું?",
                        "occasion": "કયા અવસર માટે બતાવું?",
                    }
                else:
                    ASK_LINES = {
                        "color":    "Which color should I show?",
                        "fabric":   "Which fabric should I show?",
                        "size":     "Which size should I show?",
                        "occasion": "Which occasion should I show?",
                    }
                ask_line = ASK_LINES.get(ask_focus or "color", ASK_LINES["color"])
                reply_text = f"{reply_text}\n\n" + "\n".join(suggested_parts) + f"\n{ask_line}"
            else:
                # Fallback to tenant categories if no attribute suggestions could be generated
                try:
                    async with SessionLocal() as db:
                        # If we have a type filter, get categories for that type
                        if requested_type:
                            cats = await get_categories_by_type(db, tenant_id, requested_type)
                        else:
                            cats = await resolve_categories(db, tenant_id, {})
                except Exception:
                    cats = []
                if cats:
                    bullets = "\n".join(f"• {c}" for c in cats[:12])
                    if lr.startswith("hi"):
                        reply_text += f"\n\nशायद आप इन कैटेगरी को देखना चाहेंगे:\n{bullets}"
                    elif lr.startswith("gu"):
                        reply_text += f"\n\nતમે આ કેટેગરી પસંદ કરો:\n{bullets}"
                    else:
                        reply_text += f"\n\nYou can try these categories:\n{bullets}"
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
        # =========================
        # SUCCESS CASE (HAVE PRODUCTS)
        # =========================
        # Collect images (max 4)
        seen, image_urls = set(), []
        for p in (pinecone_data or []):
            for u in (p.get("image_urls") or []):
                if u and u not in seen:
                    seen.add(u)
                    image_urls.append(u)
        image_urls = image_urls[:4]
        # Heading: "Here are rental saree for wedding:"
        cat_disp = str((acc_entities.get("category") or filtered_entities.get("category") or "product")).strip().lower()
        occ_disp = str((acc_entities.get("occasion") or filtered_entities.get("occasion") or "")).strip().lower()
        is_rental = bool((acc_entities.get("is_rental")
                        if acc_entities.get("is_rental") is not None
                        else filtered_entities.get("is_rental")))
        rental_word = "rental " if is_rental else ""
        heading = f"Here are {rental_word}{cat_disp} for {occ_disp}:".strip().replace("  ", " ")
        # Build the list in your desired format
        collected_for_text = {
            k: v for k, v in (filtered_entities or {}).items()
            if k in ("category", "color", "fabric", "size", "is_rental", "occasion") and v not in (None, "", [], {})
        }
        product_lines = []
        for product in (pinecone_data or []):
            name = product.get("name") or product.get("product_name") or "Unnamed Product"
            tags = _build_item_tags(product, collected_for_text)  # expected to render " - rent - freesize" etc.
            url = product.get("product_url")
            if isinstance(url, str):
                url = _normalize_url(url)
            product_lines.append(f"- {name} {tags}" + (f" — {url}" if url else ""))
        products_text = f"{heading}\n" + "\n".join(product_lines)
        # Fixed follow-up text per your earlier requirement
        followup = await FollowUP_Question(intent_type, acc_entities, language, session_history=history)
        reply_text = products_text
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
            # Text reply followed by a single follow-up message from your WhatsApp sender
            history.append({"role": "assistant", "content": reply_text})
            _commit()
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
            
    # --- availability
    elif intent_type == "availability_check":
        last_main_intent_by_session[sk] = "availability_check"
        if acc_entities.get("is_rental") is False:
            ask = await FollowUP_Question(
                intent_type,
                acc_entities,
                language,
                session_history=history,
                only_fields=["confirmation", "quantity", "price"],  # purchase-relevant asks
                max_fields=1
            )
            history.append({"role": "assistant", "content": ask}); _commit()
            return {
                "input_text": text,
                "language": language,
                "intent_type": intent_type,
                "history": history,
                "collected_entities": acc_entities,
                "reply": ask,
                "reply_text": ask
            }
        # For rentals: enforce quantity before dates
        def _is_missing(v): 
            return v in (None, "", [], {})

        if (acc_entities.get("is_rental") is True) and _is_missing(acc_entities.get("quantity")):
            ask_q = await FollowUP_Question(
                "availability_check",   # changed
                acc_entities,
                language,
                session_history=history,
                only_fields=["quantity"],
                max_fields=1
            )
            history.append({"role": "assistant", "content": ask_q}); _commit()
            last_main_intent_by_session[sk] = "availability_check"  # added
            return {
                "input_text": text,
                "language": language,
                "intent_type": "availability_check",  # changed
                "history": history,
                "collected_entities": acc_entities,
                "reply": ask_q,
                "reply_text": ask_q
            }

        # --- Define helper functions first ---
        def _p(x):
            """Robust date parse:
            1) Trust strict ISO (YYYY-MM-DD) so 2024-09-01 stays 1 Sep 2024.
            2) Otherwise, natural parse:
               - If it starts with a 4-digit year, use yearfirst=True.
               - Else prefer dayfirst=True (India)."""
            if x in (None, "", [], {}):
                return None
            s = str(x).strip()

            # 1) Strict ISO first
            try:
                return date.fromisoformat(s)
            except Exception:
                pass

            # allow 2024/09/01 or 2024.09.01 as ISO-ish
            try:
                s_isoish = re.sub(r"[./]", "-", s)
                if re.match(r"^\d{4}-\d{2}-\d{2}$", s_isoish):
                    return date.fromisoformat(s_isoish)
            except Exception:
                pass

            # 2) Natural parse with sensible flags
            try:
                starts_with_year = bool(re.match(r"^\s*\d{4}\D", s))
                dt = dateparser.parse(
                    s,
                    dayfirst=not starts_with_year,   # "1/9" → dayfirst; "2024-09-01" → yearfirst
                    yearfirst=starts_with_year,
                    fuzzy=True,
                    default=datetime.now(IST)
                )
                return dt.date() if dt else None
            except Exception:
                return None

        def _has_year(s):
            return bool(re.search(r"\b(19|20)\d{2}\b", str(s or "")))

        # --- 0) snapshot from BEFORE merge (so we can preserve prior start_date)
        prev_start_raw = (prev_entities or {}).get("start_date")
        prev_end_raw   = (prev_entities or {}).get("end_date")
        prev_start = _p(prev_start_raw)
        prev_end   = _p(prev_end_raw)

        # --- 1) what THIS turn extracted (raw, before merge)
        turn_start_raw = (raw_new_entities or {}).get("start_date")
        turn_end_raw   = (raw_new_entities or {}).get("end_date")
        turn_start = _p(turn_start_raw)
        turn_end   = _p(turn_end_raw)

        msg = (text or "")
        msg_lower = msg.lower()
        dash_is_range = re.search(r"\b\d{1,2}\s*[-–—]\s*\d{1,2}\b", msg) is not None  # only hyphen with spaces around
        has_range_tokens = (
            re.search(r"\b(to|till|until|upto|up to|between|from)\b", msg_lower) is not None
            or (" se " in msg_lower and " tak " in msg_lower)  # Hindi "se ... tak"
            or dash_is_range
        )
        # Did the USER actually type a year in the message (ignore LLM-normalized years)
        msg_has_year = bool(re.search(r"\b(19|20)\d{2}\b", msg))

        turn_has_both_distinct = bool(turn_start and turn_end and turn_start != turn_end)
        turn_is_single = (bool(turn_start) ^ bool(turn_end)) or (turn_start and turn_end and turn_start == turn_end)

        # --- 2) decide start/end for this request (IGNORE merged memory for this step)
        start_date = prev_start
        end_date   = prev_end

        # Case A: we already had a start, user sent ONE date now -> treat as END
        if prev_start and not prev_end and turn_is_single and not has_range_tokens:
            start_date = prev_start  # Keep previous start date - CRITICAL
            end_date = (turn_end or turn_start)  # Use new date as end date

            # Align based on whether the USER typed a year (message), not LLM's normalized string
            if end_date and not msg_has_year:
                try:
                    candidate_end = end_date.replace(year=start_date.year)
                    if candidate_end < start_date:
                        candidate_end = candidate_end.replace(year=start_date.year + 1)
                    end_date = candidate_end
                except Exception:
                    pass

            # Update memory immediately so fallback can't override
            acc_entities["start_date"] = start_date.isoformat()
            acc_entities["end_date"] = end_date.isoformat()

        # Case B: no previous start; user sent a single date (not a range)
        elif not prev_start and turn_is_single and (turn_start is not None) and not has_range_tokens:
            # Parse to a date object safely
            start_date = _parse_as_date(turn_start)

            if start_date and not msg_has_year:
                today = datetime.now(IST).date()
                if not _has_month(msg):
                    # Snap to current month & year
                    y, m = today.year, today.month
                    last = calendar.monthrange(y, m)[1]
                    d = min(start_date.day, last)
                    start_date = date(y, m, d)
                else:
                    # Month was typed — keep month, snap only year (and bump if still past)
                    candidate = start_date.replace(year=today.year)
                    if candidate < today:
                        candidate = candidate.replace(year=today.year + 1)
                    start_date = candidate

            end_date = None

            if start_date:
                acc_entities["start_date"] = start_date.isoformat()
            else:
                # parsing failed; let followup ask for a valid date
                acc_entities.pop("start_date", None)

            # critical: wipe any accidental single-day 'end_date' coming from NER/merge
            acc_entities.pop("end_date", None)
        elif prev_start and prev_end and turn_is_single and (turn_start or turn_end) and not has_range_tokens:
            start_date = _parse_as_date(turn_start or turn_end)
            end_date = None

            if start_date and not msg_has_year:
                today = datetime.now(IST).date()
                if not _has_month(msg):
                    # Month not typed → snap to current month/year (cap day)
                    y, m = today.year, today.month
                    last = calendar.monthrange(y, m)[1]
                    start_date = date(y, m, min(start_date.day, last))
                else:
                    # Month typed → keep month/day; snap year (bump if still past)
                    candidate = start_date.replace(year=today.year)
                    if candidate < today:
                        candidate = candidate.replace(year=today.year + 1)
                    start_date = candidate

        if (start_date is not None) and (end_date is None):
            # Single-date turn → keep start, clear end (we’ll ask for it)
            acc_entities["start_date"] = start_date.isoformat()
            acc_entities.pop("end_date", None)
        elif (start_date is not None) and (end_date is not None):
            # Full range is known this turn → persist both
            acc_entities["start_date"] = start_date.isoformat()
            acc_entities["end_date"]   = end_date.isoformat()
        # Case C: explicit range this turn (or two different dates)
        elif (turn_start is not None and turn_end is not None) and (turn_has_both_distinct or has_range_tokens):
            # Parse both into date objects
            s = _parse_as_date(turn_start)
            e = _parse_as_date(turn_end)

            if s and e:
                start_date, end_date = s, e

                if not msg_has_year:
                    today = datetime.now(IST).date()

                    if not _has_month(msg):
                        # Snap BOTH to current YEAR & MONTH
                        y, m = today.year, today.month
                        last_this = calendar.monthrange(y, m)[1]
                        d1 = min(start_date.day, last_this)
                        d2 = min(end_date.day,   last_this)
                        s_aligned = date(y, m, d1)

                        # If end day < start day in the same month, push end to next month safely
                        if d2 < d1:
                            nm, ny = (m + 1, y) if m < 12 else (1, y + 1)
                            last_next = calendar.monthrange(ny, nm)[1]
                            e_aligned = date(ny, nm, min(end_date.day, last_next))
                        else:
                            e_aligned = date(y, m, d2)
                    else:
                        # Month was typed — keep month, snap only year
                        s_aligned = date(today.year, start_date.month, start_date.day)
                        e_aligned = date(today.year, end_date.month,   end_date.day)

                    # If the aligned range is entirely in the past, prompt for future dates (existing behavior)
                    if e_aligned < today:
                        msg = _past_date_msg(s_aligned, e_aligned, language)
                        history.append({"role": "assistant", "content": msg}); _commit()
                        return {
                            "input_text": text, "language": language, "intent_type": intent_type,
                            "history": history, "collected_entities": acc_entities,
                            "reply": msg, "reply_text": msg
                        }

                    start_date, end_date = s_aligned, e_aligned

                acc_entities["start_date"] = start_date.isoformat()
                acc_entities["end_date"]   = end_date.isoformat()
            else:
                # couldn't parse; let follow-up ask for dates
                acc_entities.pop("start_date", None)
                acc_entities.pop("end_date", None)
        logging.info(
            f"[DATES ALIGNED] msg_has_year={msg_has_year}, has_range_tokens={has_range_tokens}, "
            f"start_date={start_date}, end_date={end_date}"
        )
        logging.info(f"[ENTITIES AFTER ALIGNMENT] {acc_entities}")

        # --- 3) if still missing, *then* fall back to memory
        if start_date is None and acc_entities.get("start_date"):
            start_date = _p(acc_entities.get("start_date"))
        if end_date is None and acc_entities.get("end_date"):
            end_date = _p(acc_entities.get("end_date"))

        # --- 4) ask for missing piece(s)
        if start_date is None:
            ask = await FollowUP_Question(
                intent_type, acc_entities, language, session_history=history,
                only_fields=["start_date"], max_fields=1
            )
            history.append({"role": "assistant", "content": ask}); _commit()
            return {
                "input_text": text, "language": language, "intent_type": intent_type,
                "history": history, "collected_entities": acc_entities,
                "reply": ask, "reply_text": ask
            }

        if end_date is None:
            ask = await FollowUP_Question(
                intent_type, acc_entities, language, session_history=history,
                only_fields=["end_date"], max_fields=1
            )
            history.append({"role": "assistant", "content": ask}); _commit()
            return {
                "input_text": text, "language": language, "intent_type": intent_type,
                "history": history, "collected_entities": acc_entities,
                "reply": ask, "reply_text": ask
            }

        # --- 5) sanity: end >= start (try message-based alignment first, then ask)
        if end_date < start_date:
            # If the user didn't type a year, assume it's day+month and align to start's year (or next)
            if not msg_has_year:
                try:
                    candidate_end = end_date.replace(year=start_date.year)
                    if candidate_end < start_date:
                        candidate_end = candidate_end.replace(year=start_date.year + 1)
                    end_date = candidate_end
                    acc_entities["end_date"] = end_date.isoformat()
                except Exception:
                    pass

            # If still invalid, clear end_date so the follow-up actually asks for it (no "Thank you...")
            if end_date < start_date:
                acc_entities.pop("end_date", None)
                warn = f"End date cannot be before the start date ({start_date.strftime('%d %b %Y')})."
                ask = await FollowUP_Question(
                    intent_type, acc_entities, language, session_history=history,
                    only_fields=["end_date"], max_fields=1
                )
                combined = f"{warn} {ask}".strip()
                history.append({"role": "assistant", "content": combined}); _commit()
                return {
                    "input_text": text, "language": language, "intent_type": intent_type,
                    "history": history, "collected_entities": acc_entities,
                    "reply": combined, "reply_text": combined
                }

        # --- 6) Auto-bump year without collapsing the range (only if not handled above)
        any_year_specified = (
            _has_year(prev_start_raw) or _has_year(prev_end_raw) or
            _has_year(turn_start_raw) or _has_year(turn_end_raw)
        )
        typed_range_this_turn = bool(turn_has_both_distinct or (turn_start and turn_end and has_range_tokens))
        if start_date and not any_year_specified and not typed_range_this_turn:
            today = _date.today()
            print("today=",today)
            current_year = _date.today().year
            print("current_year=",current_year)
            if start_date < today:
                try:
                    original_delta = (end_date - start_date) if end_date else None

                    # Bring start into this year; if still past, push to next year
                    candidate = start_date.replace(year=today.year)
                    if candidate < today:
                        candidate = candidate.replace(year=today.year + 1)
                    start_date = candidate

                    if end_date:
                        # Only shift end if the user never stated a year for it
                        if not _has_year(turn_end_raw) and not _has_year(prev_end_raw):
                            if original_delta is not None:
                                end_date = start_date + original_delta
                            else:
                                # align to start's year if no delta
                                end_date = end_date.replace(year=start_date.year)

                        # Guard: keep range valid without snapping to same day
                        if end_date < start_date:
                            try:
                                end_date = end_date.replace(year=end_date.year + 1)
                            except Exception:
                                # last-resort: preserve at least a 1-day range
                                end_date = start_date + (original_delta or (end_date - end_date))

                    acc_entities["start_date"] = start_date.isoformat()
                    if end_date:
                        acc_entities["end_date"] = end_date.isoformat()
                except Exception:
                    pass

        # --- 7) ensure we have a variant
        variant_id = acc_entities.get("product_variant_id")
        if not variant_id:
            resolve_filters = {
                k: v for k, v in acc_entities.items()
                if k in ("category", "color", "fabric", "size", "is_rental", "occasion")
                and v not in (None, "", [], {})
            }
            try:
                matches = await pinecone_fetch_records(resolve_filters, tenant_id)
            except Exception:
                matches = []

            def _match_attr(m, key):
                return (str(m.get(key) or "").strip().lower()
                        == str(resolve_filters.get(key) or "").strip().lower()) if resolve_filters.get(key) else True

            strict = [m for m in (matches or []) if _match_attr(m, "color") and _match_attr(m, "size") and _match_attr(m, "fabric")]
            picked = None
            if len(strict) == 1:
                picked = strict[0]
            elif len(matches or []) == 1:
                picked = matches[0]

            if picked and picked.get("variant_id"):
                variant_id = picked["variant_id"]
                acc_entities["product_variant_id"] = variant_id
            else:
                opts = (matches or [])[:5]
                if not opts:
                    reply = "Sorry I Don't Get This Product"
                else:
                    nums = "\n".join(
                        f"{i+1}) {o.get('name') or 'Item'} — {o.get('color') or '-'} | {o.get('size') or '-'} | {o.get('fabric') or '-'}"
                        for i, o in enumerate(opts)
                    )
                    reply = f"Please Choose Any One:\n{nums}\n\n1–{len(opts)}"
                history.append({"role": "assistant", "content": reply}); _commit()
                return {
                    "input_text": text, "language": language, "intent_type": intent_type,
                    "history": history, "collected_entities": acc_entities,
                    "reply": reply, "reply_text": reply,
                }

        # --- 8) DB check (both dates present now)
        try:
            async with SessionLocal() as db:
                available = await is_variant_available(db, int(float(variant_id)), start_date, end_date)
        except Exception:
            logging.exception("Availability DB check failed")
            err = "❌ I hit a server error while checking availability. Please try again."
            history.append({"role": "assistant", "content": err}); _commit()
            return {
                "input_text": text, "language": language, "intent_type": intent_type,
                "history": history, "collected_entities": acc_entities,
                "reply": err, "reply_text": err,
            }

        # --- 9) reply + next-step (confirm only if available; else ask new dates)
        lr = (language or "en-IN").split("-")[0].lower()

        if available:
            avail_line = (
                f"✅ Available {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}."
            )
            followup_line = await FollowUP_Question(
                intent_type, acc_entities, language, session_history=history,
                only_fields=["confirmation"], max_fields=1
            )
        else:
            avail_line = (
                f"❌ Not available {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}."
            )
            # Localized ask for another date range (do NOT ask for confirmation)
            if lr == "hi":
                followup_line = "कृपया कोई दूसरा तारीख़ रेंज साझा करें."
            elif lr == "gu":
                followup_line = "કૃપા કરીને બીજી તારીખો શેર કરો."
            else:
                followup_line = "Please share another date range."

        final_reply = avail_line.strip()

        history.append({"role": "assistant", "content": final_reply}); _commit()
        payload = {
            "input_text": text,
            "language": language,
            "intent_type": intent_type,
            "history": history,
            "collected_entities": acc_entities,
            "reply": final_reply,
            "reply_text": final_reply,
            "followup_reply": followup_line
        }
        if mode == "call":
            payload["answer"] = final_reply
        return payload



    elif intent_type == "website_inquiry":
        print("="*20)
        print(f"Detected Entites in analyze_message: {new_entities}")
        print("="*20)

        filtered_entities = filter_non_empty_entities(new_entities)
        print("="*20)
        print(f"Filtered Entites in analyze_message: {filtered_entities}")
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
            # Check if this is a type-based category request (e.g., "products for men")
            requested_type = acc_entities.get("type")
            asking_categories = requested_type and not any(acc_entities.get(k) for k in ["category", "color", "fabric", "size"])
            
            if requested_type and asking_categories:
                # Handle type-based category request
                try:
                    categories = await get_categories_by_type(session, tenant_id, requested_type)
                    
                    if categories:
                        lr = (language or "en-IN").split("-")[0].lower()
                        bullets = "\n".join(f"• {c}" for c in categories[:12])
                        
                        if lr == "hi":
                            reply_text = f"{requested_type.title()} के लिए उपलब्ध कैटेगरी:\n{bullets}\nकिस कैटेगरी में देखना चाहेंगे?"
                        elif lr == "gu":
                            reply_text = f"{requested_type.title()} માટે ઉપલબ્ધ કેટેગરી:\n{bullets}\nકઈ કેટેગરીમાં જોવું છે?"
                        else:
                            reply_text = f"Available categories for {requested_type}:\n{bullets}\nWhich category would you like to explore?"
                    else:
                        lr = (language or "en-IN").split("-")[0].lower()
                        if lr == "hi":
                            reply_text = f"खुशी, {requested_type} के लिए फिलहाल कोई प्रोडक्ट उपलब्ध नहीं है."
                        elif lr == "gu":
                            reply_text = f"માફ કરશો, {requested_type} માટે હાલમાં કોઈ પ્રોડક્ટ ઉપલબ્ધ નથી."
                        else:
                            reply_text = f"Sorry, no products available for {requested_type} currently."
                            
                except Exception as e:
                    logging.error(f"Error in asking_inquiry type-based categories: {e}")
                    # Fallback to regular inquiry handling
                    reply_text = await handle_asking_inquiry_variants(
                        text=text,
                        acc_entities=acc_entities or {},
                        db=session,
                        tenant_id=tenant_id,
                        detect_requested_attributes_async=detect_requested_attributes_async,
                        language=language,
                    )
            else:
                # Regular inquiry handling
                reply_text = await handle_asking_inquiry_variants(
                    text=text,
                    acc_entities=acc_entities or {},
                    db=session,
                    tenant_id=tenant_id,
                    detect_requested_attributes_async=detect_requested_attributes_async,
                    language=language,
                )

        # ADD THESE LINES - MISSING RETURN STATEMENTS
        print("reply_text=", reply_text)
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

    elif intent_type == "confirmation":
        # LLM already decided this is a confirmation.
        # Just render a single-line summary and STOP (no follow-ups).

        def _fmt(d):
            if not d:
                return None
            obj = _parse_as_date(d)
            return obj.strftime("%d %b %Y") if obj else None

        name = (acc_entities.get("name") or "your selection")
        qty  = acc_entities.get("quantity") or 1
        rb   = "rental" if acc_entities.get("is_rental") else "purchase"

        s = _fmt(acc_entities.get("start_date"))
        e = _fmt(acc_entities.get("end_date"))
        # show dates only for rental and only if both exist
        date_part = f" | {s} to {e}" if (acc_entities.get("is_rental") and s and e) else ""

        reply = f"Thanks for confirming. Summary: {name} | qty {qty} | {rb}{date_part}."
        # mark confirmed so future 'yes' replies don’t trigger more prompts
        acc_entities["order_confirmed"] = True

        try:
            if acc_entities.get("is_rental") is True:
                rid = await _create_rental_if_needed(acc_entities)
                if rid:
                    reply += f" | Rental ID: #{rid}"
        except Exception as _e:
            # keep flow resilient; don’t crash user chat if insert fails
            logging.exception("Rental insert failed in confirmation: %s", _e)

        history.append({"role": "assistant", "content": reply})
        session_memory[sk] = history
        session_entities[sk] = acc_entities

        return {
            "input_text": text,
            "language": language,
            "intent_type": "confirmation",
            "history": history,
            "collected_entities": acc_entities,
            "reply_text": reply
            # intentionally no "followup_reply"
        }

    
    # --- other / fallback
    elif intent_type == "other" or intent_type is None:
        def _is_missing(v): 
            return v in (None, "", [], {})

        is_rental_flag = acc_entities.get("is_rental")

        # If explicitly buying, never ask rental dates.
        if is_rental_flag is False:
            in_rental_flow = False
        else:
            # If renting (or we already picked a variant and rental is not explicitly False),
            # we can still drive the rental-date flow.
            in_rental_flow = (is_rental_flag is True) or bool(acc_entities.get("product_variant_id"))

        if in_rental_flow and _is_missing(acc_entities.get("quantity")):
            ask_q = await FollowUP_Question(
                "availability_check",           # was "product_search"
                acc_entities,
                language,
                session_history=history,
                only_fields=["quantity"],
                max_fields=1
            )
            history.append({"role": "assistant", "content": ask_q}); _commit()
            last_main_intent_by_session[sk] = "availability_check"
            return {
                "input_text": text,
                "language": language,
                "intent_type": "availability_check",  # was "product_search"
                "history": history,
                "collected_entities": acc_entities,
                "reply": ask_q,
                "reply_text": ask_q
            }

        missing_dates = _is_missing(acc_entities.get("start_date")) or _is_missing(acc_entities.get("end_date"))

        if in_rental_flow and missing_dates:
            need_fields = [f for f in ("start_date", "end_date") if _is_missing(acc_entities.get(f))]
            # Ask ONLY for the missing piece(s), short and contextual
            ask = await FollowUP_Question(
                "availability_check",
                acc_entities,
                language,
                session_history=history,
                only_fields=need_fields,
                max_fields=1
            )
            history.append({"role": "assistant", "content": ask}); _commit()
            return {
                "input_text": text,
                "language": language,
                "intent_type": "availability_check",
                "history": history,
                "collected_entities": acc_entities,
                "reply": ask,
                "reply_text": ask
            }
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