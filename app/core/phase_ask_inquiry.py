
import re, difflib, logging
from typing import List, Tuple, Dict, Any
from sqlalchemy import text as sql_text
from app.db.session import SessionLocal

# ---------- DETECT WHAT USER ASKED IN THIS MESSAGE ----------
_PATTERNS = {
    "color":     re.compile(r"\b(colou?rs?|hues?|tones?|shade|shades|palette)\b", re.I),
    # FIXED: handle plurals & common synonyms
    "fabric":    re.compile(r"\b(fabrics?|materials?|cloths?|cloth|weaves?|textiles?)\b", re.I),
    "size":      re.compile(r"\b(size|sizes|measurement|measurements)\b", re.I),
    "category":  re.compile(r"\b(category|categories|type|types|kind|kinds|clothes?)\b", re.I),
    "is_rental": re.compile(r"\b(rent|rental|on\s*rent|hire)\b", re.I),
    "price":     re.compile(
    r"\b(price|prices|budget|under|below|upto|up\s*to|less\s*than|within|"
    r"above|over|greater\s*than|more\s*than|upper|starting\s*price|start(?:ing)?\s*from|price\s*starts)\b",
    re.I
),
    "occasion":  re.compile(r"\b(occasion|event|wedding|marriage|shaadi|party|festival|ceremony)\b", re.I),
}
def detect_requested_attributes(text: str) -> List[str]:
    if not text: return []
    return [a for a, rx in _PATTERNS.items() if rx.search(text or "")]

# ---------- NORMALIZERS ----------
def _norm_py(s: str) -> str:
    return re.sub(r'[\s._-]+', '', (s or '').strip().lower())

def _title_token(s: str | None) -> str | None:
    if not s: return None
    v = re.sub(r"[_-]+", " ", s)
    v = re.sub(r"\s+", " ", v).strip()
    return v.title()

# Clean filters so this turn doesn't inherit irrelevant context (e.g., old color)
_GENERIC_CATEGORY_WORDS = {"cloth", "clothes", "cloths", "category", "categories", "types", "type", "kinds", "kind"}

def clean_filters_for_turn(acc_entities: dict, asked_now: list[str]) -> dict:
    local = dict(acc_entities or {})

    # If user did NOT ask about color this turn, drop any old color filter
    if "color" not in asked_now:
        local.pop("color", None)

    # If category value is just a generic word (like "clothes"), drop it
    cat = (local.get("category") or "").strip().lower()
    if cat in _GENERIC_CATEGORY_WORDS:
        local.pop("category", None)

    return local

def expand_size_terms(sizes: list[str]) -> list[str]:
    """
    Expand common synonyms so 'free size' matches 'Free', 'Free-Size', 'One Size'.
    Keeps order + de-duplicates.
    """
    out = []
    seen = set()
    for s in sizes:
        s_clean = s.strip()
        cand = [s_clean]
        n = _norm_py(s_clean)
        if n in {"freesize", "free", "onesize"}:
            cand += ["Free Size", "Free", "One Size"]
        for c in cand:
            k = _norm_py(c)
            if k not in seen:
                seen.add(k); out.append(c)
    return out

def _split_multi(raw: Any) -> List[str]:
    """Split 'Red & Pink, Navy' => ['Red','Pink','Navy'] ; accept list/str/None."""
    if raw is None: return []
    if isinstance(raw, (list, tuple)): vals = [str(x) for x in raw]
    else: vals = [str(raw)]
    out: List[str] = []
    for v in vals:
        parts = [p.strip() for p in re.split(r"(?:,|/|&|\band\b|\bor\b)", v, flags=re.I) if p and p.strip()]
        out.extend(parts)
    # de-dup preserving order
    seen = set(); uniq = []
    for x in out:
        k = _norm_py(x)
        if k not in seen:
            seen.add(k); uniq.append(x)
    return uniq

# ---------- SCOPED FILTERS WHEN LISTING ATTRIBUTES ----------
def scope_filters_for_attr(acc_entities: dict, attr_being_listed: str, asked_now: List[str]) -> dict:
    """
    - Never self-filter on the attribute we are listing (e.g., don't filter colors when listing colors)
    - Drop stale color unless the user asked about color in THIS message
    """
    local = dict(acc_entities or {})
    local.pop(attr_being_listed, None)
    if "color" not in asked_now:
        local.pop("color", None)
    return local

# ---------- PURE-PYTHON FUZZY COLOR RESOLVER ----------
async def _fetch_candidate_colors(tenant_id: int, acc_entities: dict, limit: int = 500) -> List[str]:
    where = ["p.tenant_id = :tid", "COALESCE(pv.is_active, TRUE) = TRUE"]
    params = {"tid": tenant_id}
    # keep other filters, but not color
    if (acc_entities or {}).get("category"):
        where.append("LOWER(p.category) = LOWER(:category)")
        params["category"] = str(acc_entities["category"])
    if (acc_entities or {}).get("fabric"):
        where.append("(REGEXP_REPLACE(LOWER(pv.fabric), '[\\s._-]+', '', 'g') LIKE '%' || REGEXP_REPLACE(LOWER(:fabric), '[\\s._-]+', '', 'g') || '%')")
        params["fabric"] = str(acc_entities["fabric"])
    if (acc_entities or {}).get("size"):
        where.append("LOWER(pv.size) = LOWER(:size)")
        params["size"] = str(acc_entities["size"])

    sql = f"""
        SELECT DISTINCT TRIM(pv.color) AS color
        FROM public.products p
        JOIN public.product_variants pv ON pv.product_id = p.id
        WHERE {' AND '.join(where)} AND NULLIF(TRIM(pv.color), '') IS NOT NULL
        ORDER BY 1
        LIMIT :limit
    """
    async with SessionLocal() as db:
        rows = (await db.execute(sql_text(sql), {**params, "limit": limit})).fetchall()
    return [r[0] for r in rows if r and r[0]]

async def resolve_canonical_color(tenant_id: int, acc_entities: dict, color_query: str, threshold: float = 0.55) -> Tuple[str | None, List[str]]:
    cands = await _fetch_candidate_colors(tenant_id, acc_entities)
    if not cands: return None, []
    qn = _norm_py(color_query)
    scored = [(c, difflib.SequenceMatcher(None, qn, _norm_py(c)).ratio()) for c in cands]
    scored.sort(key=lambda x: x[1], reverse=True)
    suggestions = [c for c, _ in scored[:5]]
    best = scored[0][0] if scored and scored[0][1] >= threshold else None
    logging.info(f"[color-resolver/python] query={color_query!r} -> best={best!r}; sugg={suggestions}")
    return best, suggestions

async def resolve_color_list(tenant_id: int, acc_entities: dict, raw_colors: Any) -> Tuple[List[str], List[str]]:
    """
    Resolve a single or multi-value color field to canonical DB colors.
    Returns (resolved_colors, suggestions_pool)
    """
    colors = _split_multi(raw_colors)
    resolved: List[str] = []
    suggestions_pool: List[str] = []
    for c in colors:
        best, sugg = await resolve_canonical_color(tenant_id, acc_entities, c)
        resolved.append(best or c)
        suggestions_pool.extend(sugg or [])
    # de-dup suggestions
    seen = set(); sugg_unique = []
    for s in suggestions_pool:
        if s not in seen:
            seen.add(s); sugg_unique.append(s)
    return resolved, sugg_unique[:5]

# ---------- FETCH DISTINCT OPTIONS (color/fabric/size/category) ----------
async def fetch_distinct_options(tenant_id: int, acc_entities: dict, attr: str, limit: int = 50) -> List[str]:
    where = ["p.tenant_id = :tid", "COALESCE(pv.is_active, TRUE) = TRUE"]
    params = {"tid": tenant_id}

    if attr != "category" and (acc_entities or {}).get("category"):
        where.append("LOWER(p.category) = LOWER(:category)")
        params["category"] = str(acc_entities["category"])

    if attr != "fabric" and (acc_entities or {}).get("fabric"):
        where.append("(REGEXP_REPLACE(LOWER(pv.fabric), '[\\s._-]+', '', 'g') LIKE '%' || REGEXP_REPLACE(LOWER(:fabric), '[\\s._-]+', '', 'g') || '%')")
        params["fabric"] = str(acc_entities["fabric"])

    if attr != "color" and (acc_entities or {}).get("color"):
        where.append("(REGEXP_REPLACE(LOWER(pv.color),  '[\\s._-]+', '', 'g') LIKE '%' || REGEXP_REPLACE(LOWER(:color),  '[\\s._-]+', '', 'g') || '%')")
        params["color"] = str(acc_entities["color"])

    if attr != "size" and (acc_entities or {}).get("size"):
        where.append("LOWER(pv.size) = LOWER(:size)")
        params["size"] = str(acc_entities["size"])

    if attr == "color":   col = "pv.color"
    elif attr == "fabric":col = "pv.fabric"
    elif attr == "size":  col = "pv.size"
    elif attr == "category": col = "p.category"
    else: return []

    norm = f"LOWER(REGEXP_REPLACE({col}, '[\\s._-]+', '', 'g'))"
    sql = f"""
        SELECT MIN(TRIM({col})) AS display_value, {norm} AS norm_key, COUNT(*) AS cnt
        FROM public.products p
        JOIN public.product_variants pv ON pv.product_id = p.id
        WHERE {' AND '.join(where)} AND NULLIF(TRIM({col}), '') IS NOT NULL
        GROUP BY {norm}
        ORDER BY cnt DESC, MIN(TRIM({col})) ASC
        LIMIT :limit
    """
    async with SessionLocal() as db:
        rows = (await db.execute(sql_text(sql), {**params, "limit": limit})).fetchall()
    out = []
    for disp, _nk, _cnt in rows:
        val = _title_token(disp)
        if val: out.append(val)
    return out

# ---------- ADVANCED CATEGORY SUGGESTER (lists, ranges, booleans, synonyms) ----------
def _normalize_occasion_list(raw: Any) -> List[str]:
    occs = _split_multi(raw)
    out = []
    for o in occs:
        k = o.strip().lower()
        if k in {"marriage", "shaadi", "shaadi", "wedding ceremony"}:
            out.append("wedding")
        else:
            out.append(o)
    return out

def _coerce_bool(v: Any) -> bool | None:
    if v is None: return None
    if isinstance(v, bool): return v
    s = str(v).strip().lower()
    if s in {"1","true","yes","y","rent","rental","on rent"}: return True
    if s in {"0","false","no","n","buy","purchase"}: return False
    return None

def _price_bounds(acc: dict, key: str) -> Tuple[float | None, float | None]:
    """
    Interpret a price value in entities:
    - If only one number is present (e.g., 'under 5000'), treat as MAX.
    - If you store min/max separately, respect 'price_min'/'price_max' keys when present.
    """
    pmin = acc.get(f"{key}_min"); pmax = acc.get(f"{key}_max")
    pv = acc.get(key)
    try:
        if pmin is not None: pmin = float(pmin)
        if pmax is not None: pmax = float(pmax)
        if pv is not None:
            # treat single 'price' as max/budget
            val = float(pv)
            if pmax is None: pmax = val
    except Exception:
        pass
    return pmin, pmax

async def suggest_categories_advanced(tenant_id: int, acc_entities: dict) -> List[str]:
    """
    Supports:
    - colors: one or many (['Red','Pink']) with fuzzy resolution
    - fabrics: one or many (['Silk','Georgette'])
    - sizes: one or many (['Large','Medium'])
    - occasion: list with synonyms normalized (wedding/marriage/shaadi)
    - is_rental: boolean
    - price / rental_price: min/max (single number treated as MAX)
    Returns distinct category names ordered by popularity.
    """
    local = dict(acc_entities or {})

    # Resolve multi colors (fuzzy)
    colors_raw = _split_multi(local.get("color"))
    if colors_raw:
        resolved, _sugg = await resolve_color_list(tenant_id, local, colors_raw)
        local["color__list"] = resolved

    # Normalize lists
    fabrics = _split_multi(local.get("fabric"))
    sizes   = _split_multi(local.get("size"))
    occs    = _normalize_occasion_list(local.get("occasion"))
    is_rent = _coerce_bool(local.get("is_rental"))

    price_min, price_max           = _price_bounds(local, "price")
    rental_price_min, rental_price_max = _price_bounds(local, "rental_price")

    where = ["p.tenant_id = :tid", "COALESCE(pv.is_active, TRUE) = TRUE"]
    params: Dict[str, Any] = {"tid": tenant_id}

    # Colors (OR across provided colors)
    if local.get("color__list"):
        ors = []
        for idx, c in enumerate(local["color__list"]):
            key = f"col_{idx}"
            ors.append(
    f"REGEXP_REPLACE(LOWER(pv.color), '[\\s._-]+', '', 'g') "
    f"LIKE '%%' || REGEXP_REPLACE(LOWER(:{key}), '[\\s._-]+', '', 'g') || '%%'"
)
            params[key] = c
        where.append("(" + " OR ".join(ors) + ")")

    # Fabrics (OR)
    if fabrics:
        ors = []
        for idx, f in enumerate(fabrics):
            key = f"fab_{idx}"
            ors.append(
    f"REGEXP_REPLACE(LOWER(pv.fabric), '[\\s._-]+', '', 'g') "
    f"LIKE '%%' || REGEXP_REPLACE(LOWER(:{key}), '[\\s._-]+', '', 'g') || '%%'"
)
            params[key] = f
        where.append("(" + " OR ".join(ors) + ")")

    # Sizes (IN)
    sizes = expand_size_terms(_split_multi(local.get("size")))
    if sizes:
        ors = []
        for idx, s in enumerate(sizes):
            key = f"sz_{idx}"
            ors.append(
                f"REGEXP_REPLACE(LOWER(pv.size), '[\\s._-]+', '', 'g') "
                f"LIKE '%%' || REGEXP_REPLACE(LOWER(:{key}), '[\\s._-]+', '', 'g') || '%%'"
            )
            params[key] = s
        where.append("(" + " OR ".join(ors) + ")")

        # Rent or Buy
        if is_rent is not None:
            where.append("pv.is_rental = :is_rental")
            params["is_rental"] = is_rent

    # Price bounds (purchase)
    if price_min is not None:
        where.append("pv.price IS NOT NULL AND pv.price >= :pmin")
        params["pmin"] = price_min
    if price_max is not None:
        where.append("pv.price IS NOT NULL AND pv.price <= :pmax")
        params["pmax"] = price_max

    # Rental price bounds
    if rental_price_min is not None:
        where.append("pv.rental_price IS NOT NULL AND pv.rental_price >= :rpmin")
        params["rpmin"] = rental_price_min
    if rental_price_max is not None:
        where.append("pv.rental_price IS NOT NULL AND pv.rental_price <= :rpmax")
        params["rpmax"] = rental_price_max

    sql = f"""
        SELECT p.category, COUNT(*) AS cnt
        FROM public.products p
        JOIN public.product_variants pv ON pv.product_id = p.id
        WHERE {' AND '.join(where)}
        GROUP BY p.category
        HAVING p.category IS NOT NULL AND p.category <> ''
        ORDER BY cnt DESC, p.category ASC
        LIMIT 50
    """
    async with SessionLocal() as db:
        rows = (await db.execute(sql_text(sql), params)).fetchall()
    return [r[0] for r in rows if r and r[0]]

def _price_mode_from_text_and_entities(text: str, acc_entities: dict) -> str:
    t = (text or "").lower()
    if "rent" in t or "rental" in t:
        return "rental_price"
    if str((acc_entities or {}).get("is_rental")).lower() in {"1", "true", "yes"}:
        return "rental_price"
    return "price"  # default: purchase price

_PRICE_NUM = re.compile(r"(\d{2,7})")  # 2..7 digits to catch 99–9,999,999

def parse_price_phrase(text: str) -> dict:
    """
    Returns dict with any of:
      {
        "starting": True|False,
        "min": float|None,
        "max": float|None
      }
    Examples:
      "under 1000"          -> {"max": 1000}
      "below 1500 rupees"   -> {"max": 1500}
      "upto 2500"           -> {"max": 2500}
      "above 2000"          -> {"min": 2000}
      "upper 2000 rupees"   -> {"min": 2000}
      "over 3000"           -> {"min": 3000}
      "between 1200 and 2500" -> {"min": 1200, "max": 2500}
      "starting price of saree" -> {"starting": True}
      "what's the starting price" -> {"starting": True}
    """
    s = (text or "").lower()

    # explicit "between X and Y" / "X - Y"
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

    # "starting price" (may or may not include a number)
    if re.search(r"\b(start(?:ing)?\s*price|start(?:ing)?\s*from|price\s*starts)\b", s):
        # If a number is present, treat it as a min (rare wording like "starting from 999")
        m = _PRICE_NUM.search(s)
        if m:
            return {"starting": True, "min": float(m.group(1))}
        return {"starting": True}

    # bare number phrase (e.g., "under 1000" mis-tagged, or "budget 1500")
    # If we only see one number with no qualifiers, treat it as MAX budget.
    nums = _PRICE_NUM.findall(s)
    if len(nums) == 1:
        return {"max": float(nums[0])}

    # no usable pricing phrase
    return {}

def apply_price_parsing_to_entities(text: str, acc_entities: dict) -> dict:
    """
    Reads 'price' from text (and rental intent) and writes numeric bounds into acc_entities:
      - price_min / price_max   OR
      - rental_price_min / rental_price_max
    Also sets a private flag "__starting_price__" when the query is "starting price".
    Returns a new dict; does not mutate input.
    """
    out = dict(acc_entities or {})
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

def _where_from_entities_for_price(entities: dict, price_col: str) -> tuple[list[str], dict]:
    """
    Build WHERE for starting-price queries using your relaxed match rules.
    NOTE: Do NOT add price bounds here; we want the true minimum.
    """
    e = entities or {}
    where = ["p.tenant_id = :tid", "COALESCE(pv.is_active, TRUE) = TRUE", f"{price_col} IS NOT NULL"]
    params = {}

    if e.get("category"):
        where.append("LOWER(p.category) = LOWER(:category)")
        params["category"] = str(e["category"])

    if e.get("fabric"):
        where.append(
            "(REGEXP_REPLACE(LOWER(pv.fabric), '[\\s._-]+', '', 'g') "
            "LIKE '%%' || REGEXP_REPLACE(LOWER(:fabric), '[\\s._-]+', '', 'g') || '%%')"
        )
        params["fabric"] = str(e["fabric"])

    if e.get("color"):
        where.append(
            "(REGEXP_REPLACE(LOWER(pv.color), '[\\s._-]+', '', 'g') "
            "LIKE '%%' || REGEXP_REPLACE(LOWER(:color), '[\\s._-]+', '', 'g') || '%%')"
        )
        params["color"] = str(e["color"])

    if e.get("size"):
        where.append("LOWER(pv.size) = LOWER(:size)")
        params["size"] = str(e["size"])

    if e.get("occasion"):
        where.append("p.occasion ILIKE :occ_like")
        params["occ_like"] = f"%{e['occasion']}%"

    # respect is_rental flag if provided (but not mandatory for purchase mode)
    if e.get("is_rental") is not None:
        where.append("pv.is_rental = :is_rental")
        params["is_rental"] = bool(e["is_rental"])

    return where, params

async def fetch_starting_price(tenant_id: int, acc_entities: dict, mode: str = "price"):
    """
    mode: 'price' or 'rental_price'
    If category present -> returns {'category': cat, 'value': min_value}
    Else -> returns list[{'category': cat, 'value': min_value}] for top categories by min price (ascending).
    """
    price_col = "pv.price" if mode == "price" else "pv.rental_price"
    where, params = _where_from_entities_for_price(acc_entities, price_col)
    params["tid"] = tenant_id

    async with SessionLocal() as db:
        if (acc_entities or {}).get("category"):
            sql = f"""
                SELECT MIN({price_col}) AS minv
                FROM public.products p
                JOIN public.product_variants pv ON pv.product_id = p.id
                WHERE {' AND '.join(where)}
            """
            row = (await db.execute(sql_text(sql), params)).fetchone()
            v = row[0] if row else None
            return {"category": acc_entities.get("category"), "value": float(v) if v is not None else None}
        else:
            # starting price per category (cheapest first)
            sql = f"""
                SELECT p.category, MIN({price_col}) AS minv
                FROM public.products p
                JOIN public.product_variants pv ON pv.product_id = p.id
                WHERE {' AND '.join(where)}
                GROUP BY p.category
                HAVING p.category IS NOT NULL AND p.category <> ''
                ORDER BY MIN({price_col}) ASC, p.category ASC
                LIMIT 12
            """
            rows = (await db.execute(sql_text(sql), params)).fetchall()
            return [{"category": r[0], "value": float(r[1])} for r in rows if r and r[1] is not None]

def _fmt_rs(v: float | None) -> str:
    return f"₹{int(round(v))}" if v is not None else "—"

def render_starting_price_single(lang_root: str, cat: str, v: float | None, mode: str) -> str:
    label = "rental" if mode == "rental_price" else "price"
    title = {
        "en": f"Starting {label} for {cat}",
        "hi": f"{cat} की शुरुआती {('किराया' if mode=='rental_price' else 'कीमत')}",
        "gu": f"{cat} માટેની શરૂઆતની {('ભાડું' if mode=='rental_price' else 'કિંમત')}",
    }.get(lang_root, f"Starting {label} for {cat}")
    return f"{title}: {_fmt_rs(v)}"

def render_starting_price_table(lang_root: str, rows: list[dict], mode: str) -> str:
    label = "rental" if mode == "rental_price" else "price"
    title = {"en": f"Starting {label} by category",
             "hi": f"श्रेणी अनुसार शुरुआती {('किराया' if mode=='rental_price' else 'कीमत')}",
             "gu": f"કેટેગરી પ્રમાણે શરૂઆતની {('ભાડું' if mode=='rental_price' else 'કિંમત')}"}.get(lang_root, f"Starting {label} by category")
    body = "\n".join(f"• {r['category']}: {_fmt_rs(r['value'])}" for r in rows[:12])
    return f"{title}:\n{body}"

# ---------- RENDER HELPERS ----------
def render_attr_options(lang_root: str, attr: str, options: List[str]) -> str:
    title_map = {
        "en": {"color": "Available colors", "fabric": "Available fabrics", "size": "Available sizes", "category": "Available categories"},
        "hi": {"color": "उपलब्ध रंग",       "fabric": "उपलब्ध फैब्रिक",   "size": "उपलब्ध साइज",   "category": "उपलब्ध कैटेगरी"},
        "gu": {"color": "ઉપલબ્ધ કલર્સ",      "fabric": "ઉપલબ્ધ ફેબ્રિક",   "size": "ઉપલબ્ધ સાઇઝ",    "category": "ઉપલબ્ધ કેટેગરી"},
    }
    title = title_map.get(lang_root, title_map["en"]).get(attr, "Available options")
    lines = "\n".join(f"• {o}" for o in options[:12])
    trail = {"en": "\nWhich one would you like?", "hi": "\nकौन-सा चाहेंगे?", "gu": "\nકયું પસંદ કરશો?"}.get(lang_root, "\nWhich one would you like?")
    return f"{title}:\n{lines}{trail}"

def render_categories_reply(lang_root: str, categories: List[str]) -> str:
    title = {"en": "Available categories", "hi": "उपलब्ध कैटेगरी", "gu": "ઉપલબ્ધ કેટેગરી"}.get(lang_root, "Available categories")
    body  = "\n".join(f"• {c}" for c in categories[:12])
    trail = {"en": "\nWhich one would you like?", "hi": "\nकौन-सी देखना चाहेंगे?", "gu": "\nકઈ જોવી ગમશે?"}.get(lang_root, "\nWhich one would you like?")
    return f"{title}:\n{body}{trail}"