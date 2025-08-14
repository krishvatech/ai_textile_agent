import re
import logging
from typing import List, Tuple, Dict, Any, Iterable
from sqlalchemy import text as sql_text
from app.db.session import SessionLocal

log = logging.getLogger(__name__)

# ---------- DETECT WHAT USER ASKED IN THIS MESSAGE ----------
_PATTERNS: Dict[str, re.Pattern] = {
    "color":     re.compile(r"\b(colou?rs?|hues?|tones?|shade|shades|palette)\b", re.I),
    "fabric":    re.compile(r"\b(fabrics?|materials?|cloths?|cloth|weaves?|textiles?)\b", re.I),
    "size":      re.compile(r"\b(size|sizes|measurement|measurements)\b", re.I),
    "category":  re.compile(r"\b(category|categories|type|types|kind|kinds|clothes?)\b", re.I),
    "is_rental": re.compile(r"\b(rent|rental|on\s*rent|hire)\b", re.I),
    "price":     re.compile(
        r"\b(price|prices|budget|under|below|upto|up\s*to|less\s*than|within|"
        r"above|over|greater\s*than|more\s*than|between|starting|start|rate)\b",
        re.I,
    ),
}

def detect_requested_attributes(text: str) -> List[str]:
    if not text:
        return []
    s = text or ""
    return [a for a, rx in _PATTERNS.items() if rx.search(s)]


# ---------- PRICE PHRASE PARSER ----------
_PRICE_NUM = re.compile(r"(\d{2,7})")

def parse_price_phrase(text: str) -> Dict[str, float | bool]:
    """
    Returns any of: {"starting": bool, "min": float, "max": float}
    Handles:
      - under/below/upto/up to/less than/within X  -> {"max": X}
      - above/over/greater than/more than X        -> {"min": X}
      - between X and Y / X-Y                      -> {"min": lo, "max": hi}
      - 'starting'/'starting price'/'starting rate'/ 'rate se starting'
      - bare single number                         -> {"max": X}
    """
    s = (text or "").lower()

    # between X and Y / X - Y
    m = re.search(r"between\s*(\d{2,7})\s*(?:and|to|-)\s*(\d{2,7})", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        return {"min": lo, "max": hi}

    # under / below / upto / up to / less than / within
    if re.search(r"\b(under|below|upto|up\s*to|less\s*than|within)\b", s):
        m = _PRICE_NUM.search(s)
        if m:
            return {"max": float(m.group(1))}

    # above / over / greater than / more than / upper
    if re.search(r"\b(above|over|greater\s*than|more\s*than|upper)\b", s):
        m = _PRICE_NUM.search(s)
        if m:
            return {"min": float(m.group(1))}

    # starting variants (incl. Hinglish)
    if (
        re.search(r"\bstart(?:ing)?\b", s)
        or re.search(r"\b(start(?:ing)?\s*price|start(?:ing)?\s*from|price\s*starts|starting\s*(?:price|rate)|rate\s*se\s*starting)\b", s)
    ):
        m = _PRICE_NUM.search(s)  # e.g., "starting from 999"
        if m:
            return {"starting": True, "min": float(m.group(1))}
        return {"starting": True}

    # bare single number => budget/max
    nums = _PRICE_NUM.findall(s)
    if len(nums) == 1:
        return {"max": float(nums[0])}

    return {}


# ---------- NORMALIZATION HELPERS ----------
def _norm_py(s: str) -> str:
    """Lower, strip, collapse whitespace + punctuation to normalize text."""
    if not isinstance(s, str):
        s = str(s or "")
    s = s.strip().lower()
    s = re.sub(r"[\s\._-]+", " ", s)
    return s

def _split_multi(raw: Any) -> List[str]:
    """Split 'Red & Pink, Navy' => ['Red','Pink','Navy'] ; accept list/str/None."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        vals = [str(x) for x in raw]
    else:
        vals = [str(raw)]
    out: List[str] = []
    for v in vals:
        parts = [p.strip() for p in re.split(r"(?:,|/|&|\band\b|\bor\b)", v, flags=re.I) if p and p.strip()]
        out.extend(parts)
    # de-dup preserving order
    seen = set()
    uniq = []
    for p in out:
        k = _norm_py(p)
        if k not in seen:
            seen.add(k); uniq.append(p)
    return uniq


# ---------- OCCASION JOIN ----------
def _add_occasion_join_and_filter(
    occs: Iterable[str],
    where: List[str],
    params: Dict[str, Any],
) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    If 'occasion' or occasions present, add the join and WHERE with OR across values.
    Returns (join_sql, where, params)
    """
    occs = [str(o) for o in (_split_multi(occs) or [])]
    if not occs:
        return "", where, params

    join_sql = """
        JOIN public.product_variant_occasions pvo ON pvo.variant_id = pv.id
        JOIN public.occasions o ON o.id = pvo.occasion_id
    """

    ors = []
    for i, occ in enumerate(occs):
        key = f"occ_{i}"
        ors.append(f"LOWER(o.name) = LOWER(:{key})")
        params[key] = str(occ)
    where.append("(" + " OR ".join(ors) + ")")
    return join_sql, where, params


# ---------- COERCE RENTAL BOOLEAN ----------
def _coerce_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1","true","yes","y","rent","rental","on rent","hire"}:
        return True
    if s in {"0","false","no","n","buy","purchase","sale"}:
        return False
    return None


# ---------- PRICE BOUNDS FROM ENTITIES ----------
def _price_bounds(acc: dict, key: str) -> Tuple[float | None, float | None]:
    """
    Interpret a price value in entities:
    - If only one number is present (e.g., 'under 5000'), treat as MAX.
    - If you store min/max separately, respect 'price_min'/'price_max' keys when present.
    Returns (min, max) for the given key: 'price' or 'rental_price'.
    """
    if key == "price":
        return acc.get("price_min"), acc.get("price_max")
    else:
        return acc.get("rental_price_min"), acc.get("rental_price_max")


# ---------- WHERE FROM ENTITIES (WITH OCCASION SUPPORT) ----------
def _where_from_entities_for_price(acc_entities: dict, price_col: str) -> Tuple[List[str], Dict[str, Any], str]:
    """
    Build WHERE pieces and params from entities for a price-oriented query.
    Returns (where_clauses, params, occ_join_sql)
    """
    acc = acc_entities or {}
    where: List[str] = ["p.tenant_id = :tid", "COALESCE(pv.is_active, TRUE) = TRUE"]
    params: Dict[str, Any] = {}

    # category
    if acc.get("category"):
        where.append("LOWER(p.category) = LOWER(:category)")
        params["category"] = str(acc["category"])

    # fabric (loose normalize)
    if acc.get("fabric"):
        where.append(
            "(REGEXP_REPLACE(LOWER(pv.fabric), '[\\s._-]+', '', 'g') "
            "LIKE '%' || REGEXP_REPLACE(LOWER(:fabric), '[\\s._-]+', '', 'g') || '%')"
        )
        params["fabric"] = str(acc["fabric"])

    # color (loose)
    if acc.get("color"):
        where.append(
            "(REGEXP_REPLACE(LOWER(pv.color), '[\\s._-]+', '', 'g') "
            "LIKE '%' || REGEXP_REPLACE(LOWER(:color), '[\\s._-]+', '', 'g') || '%')"
        )
        params["color"] = str(acc["color"])

    # size (loose)
    if acc.get("size"):
        where.append(
            "(REGEXP_REPLACE(LOWER(pv.size), '[\\s._-]+', '', 'g') "
            "LIKE '%' || REGEXP_REPLACE(LOWER(:size), '[\\s._-]+', '', 'g') || '%')"
        )
        params["size"] = str(acc["size"])

    # rental vs sale
    is_rental = _coerce_bool(acc.get("is_rental"))
    if is_rental is True:
        where.append("COALESCE(pv.is_rental, FALSE) = TRUE")
    elif is_rental is False:
        where.append("COALESCE(pv.is_rental, FALSE) = FALSE")

    # numeric bounds
    price_min, price_max = _price_bounds(acc, "price")
    rental_price_min, rental_price_max = _price_bounds(acc, "rental_price")

    if price_col.endswith(".price"):
        if price_min is not None:
            where.append("pv.price IS NOT NULL AND pv.price >= :pmin")
            params["pmin"] = float(price_min)
        if price_max is not None:
            where.append("pv.price IS NOT NULL AND pv.price <= :pmax")
            params["pmax"] = float(price_max)
    else:
        if rental_price_min is not None:
            where.append("pv.rental_price IS NOT NULL AND pv.rental_price >= :rpmin")
            params["rpmin"] = float(rental_price_min)
        if rental_price_max is not None:
            where.append("pv.rental_price IS NOT NULL AND pv.rental_price <= :rpmax")
            params["rpmax"] = float(rental_price_max)

    # occasions
    occs = acc.get("occasion") or acc.get("occasions") or []
    occ_join_sql, where, params = _add_occasion_join_and_filter(occs, where, params)

    return where, params, occ_join_sql


# ---------- CHOOSE PRICE MODE ----------
def _price_mode_from_text_and_entities(text: str, acc_entities: dict) -> str:
    """
    Decide whether the user is talking about 'price' (sale) or 'rental_price' (rent).
    Priority:
      1) Explicit is_rental in entities
      2) Text contains rent/rental/hire
      3) Default to 'price'
    """
    is_rental = _coerce_bool((acc_entities or {}).get("is_rental"))
    if is_rental is True:
        return "rental_price"
    if is_rental is False:
        return "price"
    s = (text or "").lower()
    if re.search(r"\b(rent|rental|on\s*rent|hire)\b", s):
        return "rental_price"
    return "price"


# ---------- APPLY PARSED PRICE TO ENTITIES ----------
def apply_price_parsing_to_entities(text: str, acc_entities: dict) -> dict:
    """
    Reads 'price' from text (and rental intent) and writes numeric bounds into acc_entities:
      - price_min / price_max   OR
      - rental_price_min / rental_price_max
    Also sets a private flag "__starting_price__" when the query is "starting price".
    Returns a new dict; does not mutate input.
    """
    out = dict(acc_entities or {})
    out.pop("__starting_price__", None)
    parsed = parse_price_phrase(text or "")
    if not parsed:
        return out

    mode = _price_mode_from_text_and_entities(text, out)  # 'price' or 'rental_price'
    if parsed.get("starting"):
        out["__starting_price__"] = mode   # remember which column to use for "starting"
        # If a number was present with starting, treat as lower bound
        if parsed.get("min") is not None:
            if mode == "price":
                out["price_min"] = parsed["min"]
            else:
                out["rental_price_min"] = parsed["min"]
        return out

    # Regular bounds
    if mode == "price":
        if parsed.get("min") is not None:
            out["price_min"] = parsed["min"]
        if parsed.get("max") is not None:
            out["price_max"] = parsed["max"]
    else:
        if parsed.get("min") is not None:
            out["rental_price_min"] = parsed["min"]
        if parsed.get("max") is not None:
            out["rental_price_max"] = parsed["max"]

    return out


# ---------- STARTING PRICE QUERY ----------
async def fetch_starting_price(tenant_id: int, acc_entities: dict, mode: str = "price"):
    """
    mode: 'price' or 'rental_price'
    If category present -> returns {'category': cat, 'value': min_value}
    Else -> returns list[{'category': cat, 'value': min_value}] for top categories by min price (ascending).
    """
    price_col = "pv.price" if mode == "price" else "pv.rental_price"
    where, params, occ_join_sql = _where_from_entities_for_price(acc_entities, price_col)
    params["tid"] = tenant_id

    from_sql = f"""
        FROM public.products p
        JOIN public.product_variants pv ON pv.product_id = p.id
        {occ_join_sql}
    """

    async with SessionLocal() as db:
        if (acc_entities or {}).get("category"):
            sql = f"""
                SELECT MIN({price_col}) AS minv
                {from_sql}
                WHERE {' AND '.join(where)}
            """
            row = (await db.execute(sql_text(sql), params)).fetchone()
            v = row[0] if row else None
            return {"category": acc_entities.get("category"), "value": float(v) if v is not None else None}
        else:
            sql = f"""
                SELECT p.category, MIN({price_col}) AS minv
                {from_sql}
                WHERE {' AND '.join(where)}
                GROUP BY p.category
                HAVING p.category IS NOT NULL AND p.category <> ''
                ORDER BY MIN({price_col}) ASC, p.category ASC
                LIMIT 12
            """
            rows = (await db.execute(sql_text(sql), params)).fetchall()
            return [{"category": r[0], "value": float(r[1])} for r in rows if r and r[1] is not None]


# ---------- LISTING HELPERS (colors/fabrics/sizes/categories) ----------
async def fetch_distinct_options(tenant_id: int, acc_entities: dict, attr: str, limit: int = 50) -> List[str]:
    """
    Return distinct values for a given attribute ('color'|'fabric'|'size'|'category')
    with light normalization and respecting already-selected filters on other attrs.
    """
    where = ["p.tenant_id = :tid", "COALESCE(pv.is_active, TRUE) = TRUE"]
    params: Dict[str, Any] = {"tid": tenant_id}
    occ_join_sql = ""

    # do not self-filter on the attribute being listed
    if attr != "category" and (acc_entities or {}).get("category"):
        where.append("LOWER(p.category) = LOWER(:category)")
        params["category"] = str(acc_entities["category"])

    if attr != "fabric" and (acc_entities or {}).get("fabric"):
        where.append(
            "(REGEXP_REPLACE(LOWER(pv.fabric), '[\\s._-]+', '', 'g') "
            "LIKE '%' || REGEXP_REPLACE(LOWER(:fabric), '[\\s._-]+', '', 'g') || '%')"
        )
        params["fabric"] = str(acc_entities["fabric"])

    if attr != "color" and (acc_entities or {}).get("color"):
        where.append(
            "(REGEXP_REPLACE(LOWER(pv.color), '[\\s._-]+', '', 'g') "
            "LIKE '%' || REGEXP_REPLACE(LOWER(:color), '[\\s._-]+', '', 'g') || '%')"
        )
        params["color"] = str(acc_entities["color"])

    if attr != "size" and (acc_entities or {}).get("size"):
        where.append(
            "(REGEXP_REPLACE(LOWER(pv.size), '[\\s._-]+', '', 'g') "
            "LIKE '%' || REGEXP_REPLACE(LOWER(:size), '[\\s._-]+', '', 'g') || '%')"
        )
        params["size"] = str(acc_entities["size"])

    # rental vs sale
    is_rental = _coerce_bool((acc_entities or {}).get("is_rental"))
    if is_rental is True:
        where.append("COALESCE(pv.is_rental, FALSE) = TRUE")
    elif is_rental is False:
        where.append("COALESCE(pv.is_rental, FALSE) = FALSE")

    # occasions
    occs = (acc_entities or {}).get("occasion") or (acc_entities or {}).get("occasions") or []
    occ_join_sql, where, params = _add_occasion_join_and_filter(occs, where, params)

    # Build SELECT
    col_map = {
        "category": "p.category",
        "fabric":   "pv.fabric",
        "color":    "pv.color",
        "size":     "pv.size",
    }
    sel_col = col_map.get(attr)
    if not sel_col:
        return []

    from_sql = f"""
        FROM public.products p
        JOIN public.product_variants pv ON pv.product_id = p.id
        {occ_join_sql}
    """

    sql = f"""
        SELECT {sel_col} AS v, COUNT(*) AS cnt
        {from_sql}
        WHERE {' AND '.join(where)}
        GROUP BY {sel_col}
        HAVING {sel_col} IS NOT NULL AND {sel_col} <> ''
        ORDER BY COUNT(*) DESC, {sel_col} ASC
        LIMIT :lim
    """
    params["lim"] = int(limit)

    async with SessionLocal() as db:
        rows = (await db.execute(sql_text(sql), params)).fetchall()

    # Clean + uniq while preserving order
    out: List[str] = []
    seen = set()
    for r in rows or []:
        v = str(r[0]).strip()
        if not v:
            continue
        k = _norm_py(v)
        if k not in seen:
            seen.add(k); out.append(v)
    return out


# ---------- RENDER HELPERS ----------
def _format_money(v: float | None) -> str:
    if v is None:
        return "—"
    try:
        iv = int(round(float(v)))
    except Exception:
        return str(v)
    # Simple Indian style grouping like 1,999 / 12,499
    s = f"{iv:,}"
    return s

def render_starting_price_single(lang_root: str, category: str, value: float | None, mode: str) -> str:
    label = "rental" if mode == "rental_price" else "price"
    amount = _format_money(value)
    if lang_root == "hi":
        return f"{category} का starting {label} ₹{amount} से है।"
    if lang_root == "gu":
        return f"{category} નો starting {label} ₹{amount} થી છે."
    return f"Starting {label} for {category}: ₹{amount}."

def render_starting_price_table(lang_root: str, items: List[Dict[str, Any]], mode: str) -> str:
    label = "rental" if mode == "rental_price" else "price"
    title = {
        "hi": f"श्रेणी अनुसार starting {label}:",
        "gu": f"કેટેગરી પ્રમાણે starting {label}:",
        "en": f"Starting {label} by category:",
    }.get(lang_root, f"Starting {label} by category:")
    lines = []
    for x in (items or [])[:12]:
        cat = x.get("category") or "—"
        amt = _format_money(x.get("value"))
        lines.append(f"• {cat}: ₹{amt}")
    if not lines:
        lines = ["• —"]
    return f"{title}\n" + "\n".join(lines)

def render_options_reply(lang_root: str, attr: str, options: List[str]) -> str:
    title_map = {
        "en": {"color": "Available colors", "fabric": "Available fabrics", "size": "Available sizes", "category": "Available categories"},
        "hi": {"color": "उपलब्ध रंग",       "fabric": "उपलब्ध फ़ैब्रिक",    "size": "उपलब्ध साइज",   "category": "उपलब्ध कैटेगरी"},
        "gu": {"color": "ઉપલબ્ધ કલર્સ",      "fabric": "ઉપલબ્ધ ફેબ્રિક",   "size": "ઉપલબ્ધ સાઇઝ",    "category": "ઉપલબ્ધ કેટેગરી"},
    }
    title = title_map.get(lang_root, title_map["en"]).get(attr, "Available options")
    lines = "\n".join(f"• {o}" for o in options[:12])
    trail = {"en": "\nWhich one would you like?", "hi": "\nकौन-सा पसंद करेंगे?", "gu": "\nકયું પસંદ કરશો?"}.get(lang_root, "\nWhich one would you like?")
    return f"{title}:\n{lines}{trail}"

def render_categories_reply(lang_root: str, categories: List[str]) -> str:
    title = {"en": "Available categories", "hi": "उपलब्ध कैटेगरी", "gu": "ઉપલબ્ધ કેટેગરી"}.get(lang_root, "Available categories")
    body  = "\n".join(f"• {c}" for c in categories[:12])
    trail = {"en": "\nWhich one would you like?", "hi": "\nकौन-सी देखनी है?", "gu": "\nકઈ જોવી ગમશે?"}.get(lang_root, "\nWhich one would you like?")
    return f"{title}:\n{body}{trail}"

# ----------------- ASKING/PRODUCT LISTING HELPERS -----------------

_ALLOWED_FILTER_KEYS = {
    "category", "fabric", "color", "size",
    "occasion", "occasions", "is_rental",
    "price_min", "price_max", "rental_price_min", "rental_price_max",
    "__starting_price__"
}

def _is_empty(v):
    if v is None: return True
    if isinstance(v, str) and not v.strip(): return True
    if isinstance(v, (list, tuple, dict)) and not v: return True
    return False

def _split_listish(v):
    """Split 'Red & Pink, Navy' -> ['Red','Pink','Navy'] ; pass lists through."""
    if v is None: return []
    if isinstance(v, (list, tuple)): return [str(x).strip() for x in v if str(x).strip()]
    s = str(v)
    parts = [p.strip() for p in re.split(r"(?:,|/|&|\band\b|\bor\b)", s, flags=re.I) if p and p.strip()]
    # de-dup preserving order
    seen, out = set(), []
    for p in parts:
        k = re.sub(r"[\s._-]+", " ", p.lower())
        if k not in seen:
            seen.add(k); out.append(p)
    return out

def clean_filters_for_turn(acc_entities: dict | None, asked_now: list[str] | None) -> dict:
    """
    Keep only allowed keys, normalize empties, coerce types, and unify occasion(s) => list[str].
    Does NOT drop prior sticky values — caller handles "sticky" logic.
    """
    src = dict(acc_entities or {})
    out = {}

    # pass through only allowed keys
    for k, v in src.items():
        if k not in _ALLOWED_FILTER_KEYS:
            continue
        # normalize empties like "any"/"none"
        if isinstance(v, str) and v.strip().lower() in {"any", "none", "no", "na", "n/a"}:
            v = None

        if k in {"occasion", "occasions"}:
            vals = _split_listish(v)
            if vals:
                out["occasion"] = vals  # unify to 'occasion'
            continue

        if k == "is_rental":
            s = str(v).strip().lower()
            if v is True or s in {"1","true","yes","y","rent","rental","on rent","hire"}:
                out["is_rental"] = True
            elif v is False or s in {"0","false","no","n","buy","purchase","sale"}:
                out["is_rental"] = False
            else:
                # leave unset if ambiguous
                pass
            continue

        # numeric bounds
        if k in {"price_min","price_max","rental_price_min","rental_price_max"}:
            try:
                out[k] = float(v)
            except Exception:
                pass
            continue

        # plain strings (category/fabric/color/size) incl. __starting_price__
        if not _is_empty(v):
            out[k] = v

    return out

def _format_money_inr(v):
    try:
        return f"{int(round(float(v))):,}"
    except Exception:
        return str(v)

def _build_dynamic_heading(filters: dict) -> str:
    """
    Compose a friendly heading like:
    'Here are our Kurta Sets — Rent • Wedding • (Fabric | Color | Size)'
    """
    cat = filters.get("category") or "Products"
    rent_tag = "Rent" if str(filters.get("is_rental")).lower() == "true" or filters.get("is_rental") is True else "Buy"
    chips = []

    # add high-signal facets in a stable order
    if filters.get("occasion"):
        occs = filters["occasion"] if isinstance(filters["occasion"], list) else [filters["occasion"]]
        chips.extend(occs[:2])
    if filters.get("fabric"): chips.append(str(filters["fabric"]))
    if filters.get("color"):  chips.append(str(filters["color"]))
    if filters.get("size"):   chips.append(str(filters["size"]))

    chip_txt = f" — {rent_tag}" + ((" • " + " | ".join(chips)) if chips else "")
    return f"Here are our {cat}:{chip_txt}"

def _extract_meta(item: dict) -> dict:
    """Support both Pinecone-style {'metadata': {...}} and plain dicts."""
    if not isinstance(item, dict): return {}
    return item.get("metadata") or item

def _best_name(m: dict) -> str:
    return (m.get("product_name")
            or m.get("name")
            or m.get("title")
            or (f"{m.get('category','Item')}".strip())
            or "Item")

def _detect_is_rental(m: dict, fallback: bool | None) -> bool | None:
    if m.get("is_rental") is True: return True
    if m.get("is_rental") is False: return False
    # some datasets store is_rental as 1/0 or "true"/"false"
    s = str(m.get("is_rental")).strip().lower()
    if s in {"1","true","yes"}: return True
    if s in {"0","false","no"}: return False
    return fallback

def _build_product_lines(pinecone_data: list[dict], filters: dict) -> list[str]:
    """
    Render concise list lines. Example:
      "New Partywear Jimmy Chu Saree — Rent • Jimmy Chu | Rani | Freesize"
    Falls back gracefully if fields are missing.
    """
    lines = []
    fallback_rent = filters.get("is_rental")
    for it in (pinecone_data or [])[:10]:
        m = _extract_meta(it)
        if not m: continue

        is_rent = _detect_is_rental(m, fallback_rent)
        tag = "Rent" if is_rent else "Buy"

        name = _best_name(m)
        bits = []
        if m.get("fabric"): bits.append(str(m.get("fabric")))
        if m.get("color"):  bits.append(str(m.get("color")))
        if m.get("size"):   bits.append(str(m.get("size")))

        # If you want to append price, uncomment below:
        # price = m.get("rental_price" if is_rent else "price")
        # tail_price = f" — ₹{_format_money_inr(price)}" if price not in (None, "", 0) else ""

        line = f"{name} — {tag}" + (f" • {' | '.join(bits)}" if bits else "")
        lines.append(line)
    return lines or ["(No items to show)"]
