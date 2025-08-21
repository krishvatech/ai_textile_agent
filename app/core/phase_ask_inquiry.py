from typing import Dict, List, Callable, Awaitable, Any, Optional, Iterable
import logging
from sqlalchemy import select, func, and_, or_, literal
from sqlalchemy.ext.asyncio import AsyncSession
import re

# --- import your models ---
from app.db.models import Product, ProductVariant  # adjust path if needed

# If you DO have an Occasion model via M2M, this import can stay. If not, the resolver will just skip.
try:
    from app.db.models import Occasion, product_variant_occasions
    HAS_OCCASION = True
except Exception:
    Occasion = None
    product_variant_occasions = None
    HAS_OCCASION = False


# =============== helpers ===============

def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v if str(x).strip()]
    return [str(v)]

def _add_if(attr_exists: bool, conds: List[Any], cond: Any):
    if attr_exists:
        conds.append(cond)

def _norm_nonempty(col):
    return func.nullif(func.trim(col), "")

def _add_active_trueish(model, conds: List[Any]):
    """
    Treat is_active=True or NULL as 'active' to avoid unintentionally filtering out data.
    Only adds a condition if the model has an is_active column.
    """
    if hasattr(model, "is_active"):
        col = getattr(model, "is_active")
        conds.append(or_(col.is_(True), col.is_(None)))

def _get_product_col(*candidates: str):
    """
    Return the first matching SQLAlchemy column on Product from candidate names.
    Useful when schema differs (e.g., 'category', 'category_name', 'type', etc.).
    """
    for name in candidates:
        col = getattr(Product, name, None)
        if col is not None:
            return col, name
    return None, None


async def _distinct_variant_col(
    db: AsyncSession,
    col,
    tenant_id: int,
    extra_where: Optional[Iterable[Any]] = None,
    only_in_stock: bool = True,
) -> List[str]:
    """
    Distinct non-empty values from ProductVariant.<col>, joined with Product for tenant scoping.
    """
    conds: List[Any] = [Product.id == ProductVariant.product_id, Product.tenant_id == tenant_id]

    _add_active_trueish(Product, conds)
    _add_active_trueish(ProductVariant, conds)

    if only_in_stock and hasattr(ProductVariant, "available_stock"):
        conds.append(ProductVariant.available_stock > 0)

    if extra_where:
        conds.extend(extra_where)

    val = func.trim(col).label("val")
    stmt = (
        select(val)
        .join(Product, Product.id == ProductVariant.product_id)
        .where(and_(*conds))
        .where(_norm_nonempty(col).isnot(None))
        .distinct()
        .order_by(val)  # ORDER BY expression is in the select list -> ok for DISTINCT
    )

    res = await db.execute(stmt)
    return [r[0] for r in res.all()]


async def _price_min_max(
    db: AsyncSession,
    tenant_id: int,
    extra_where: Optional[Iterable[Any]] = None,
    use_rental_price: Optional[bool] = None,
) -> List[str]:
    """
    Returns ["min – max"] for price range.

    Modes:
      • use_rental_price=True   -> rental band over variants with is_rental=True and rental_price IS NOT NULL
      • use_rental_price=False  -> purchase band over variants with is_rental=False and price IS NOT NULL.
                                   First, prefer rows where rental_price is NULL or 0 (clean purchase rows).
                                   If none found, gracefully fallback to just is_rental=False & price IS NOT NULL.
      • use_rental_price=None   -> generic purchase band (price IS NOT NULL across all variants)
    """
    def _run_minmax(conds_list):
        stmt = (
            select(func.min(target_col), func.max(target_col))
            .join(Product, Product.id == ProductVariant.product_id)
            .where(and_(*conds_list))
        )
        return stmt

    # ----- base conds -----
    base: List[Any] = [Product.id == ProductVariant.product_id, Product.tenant_id == tenant_id]
    _add_active_trueish(Product, base)
    _add_active_trueish(ProductVariant, base)

    # ----- branch logic -----
    if use_rental_price is True:
        # Rental band
        target_col = ProductVariant.rental_price
        conds = list(base)
        conds.append(ProductVariant.is_rental.is_(True))
        conds.append(ProductVariant.rental_price.isnot(None))
        if extra_where:
            conds.extend(extra_where)

        res = await db.execute(_run_minmax(conds))
        mn, mx = res.first() or (None, None)
        logging.info(f"[_price_min_max] RENTAL tenant={tenant_id} -> min={mn}, max={mx}")
        if mn is None or mx is None:
            return []
        return [f"{int(mn)} – {int(mx)}"]

    elif use_rental_price is False:
        # Purchase band — prefer rows with no rental_price set (NULL or 0), else fallback.
        target_col = ProductVariant.price

        # pass 1: clean purchase variants (is_rental=False, price not null, rental_price null/0)
        conds1 = list(base)
        conds1.append(ProductVariant.is_rental.is_(False))
        conds1.append(ProductVariant.price.isnot(None))
        conds1.append(or_(ProductVariant.rental_price.is_(None),
                          ProductVariant.rental_price == 0))
        if extra_where:
            conds1.extend(extra_where)

        res1 = await db.execute(_run_minmax(conds1))
        mn1, mx1 = res1.first() or (None, None)
        logging.info(f"[_price_min_max] PURCHASE (clean) tenant={tenant_id} -> min={mn1}, max={mx1}")

        if mn1 is not None and mx1 is not None:
            return [f"{int(mn1)} – {int(mx1)}"]

        # pass 2: relaxed purchase variants (ignore rental_price column completely)
        conds2 = list(base)
        conds2.append(ProductVariant.is_rental.is_(False))
        conds2.append(ProductVariant.price.isnot(None))
        if extra_where:
            conds2.extend(extra_where)

        res2 = await db.execute(_run_minmax(conds2))
        mn2, mx2 = res2.first() or (None, None)
        logging.info(f"[_price_min_max] PURCHASE (fallback) tenant={tenant_id} -> min={mn2}, max={mx2}")

        if mn2 is None or mx2 is None:
            return []
        return [f"{int(mn2)} – {int(mx2)}"]

    else:
        # Generic purchase price band (no is_rental filter)
        target_col = ProductVariant.price
        conds = list(base)
        conds.append(ProductVariant.price.isnot(None))
        if extra_where:
            conds.extend(extra_where)

        res = await db.execute(_run_minmax(conds))
        mn, mx = res.first() or (None, None)
        logging.info(f"[_price_min_max] GENERIC tenant={tenant_id} -> min={mn}, max={mx}")
        if mn is None or mx is None:
            return []
        return [f"{int(mn)} – {int(mx)}"]


def _filters_from_entities(entities: Dict[str, Any]) -> List[Any]:
    """
    Build WHERE conditions from already-chosen filters.
    Case-insensitive matching for category/color/size/fabric.
    Ignores 'Freesize'/'Free size'/'One Size' to avoid over-filtering.
    """
    conds: List[Any] = []

    # category (on Product) — case-insensitive
    cat_col, _cat_name = _get_product_col(
        "category", "category_name", "product_category", "category_type", "type"
    )
    if entities.get("category") and cat_col is not None:
        categories = [str(c).strip().lower() for c in _as_list(entities["category"])]
        if categories:
            conds.append(func.lower(cat_col).in_(categories))

    # color (on ProductVariant) — case-insensitive
    if entities.get("color"):
        colors = [str(c).strip().lower() for c in _as_list(entities["color"])]
        if colors:
            conds.append(func.lower(ProductVariant.color).in_(colors))

    # size (on ProductVariant) — ignore free/one-size, otherwise case-insensitive
    size_val = str((entities or {}).get("size") or "").strip().lower()
    if size_val and size_val not in ("freesize", "free size", "one size", "onesize"):
        sizes = [str(s).strip().lower() for s in _as_list(entities["size"])]
        if sizes:
            conds.append(func.lower(ProductVariant.size).in_(sizes))

    # fabric (on ProductVariant) — case-insensitive
    if entities.get("fabric"):
        fabrics = [str(f).strip().lower() for f in _as_list(entities["fabric"])]
        if fabrics:
            conds.append(func.lower(ProductVariant.fabric).in_(fabrics))

    # rental flag (only if explicitly present from NLP)
    if "is_rental" in entities and entities["is_rental"] is not None:
        conds.append(ProductVariant.is_rental.is_(bool(entities["is_rental"])))
    elif entities.get("rental") in ("Rent", "Purchase"):
        conds.append(ProductVariant.is_rental.is_(entities["rental"] == "Rent"))

    return conds


def _format_list_block(title: str, items: List[str]) -> str:
    if not items:
        return ""
    lines = "\n".join(f"• {x}" for x in items)
    return f"{title}:\n{lines}"


# =============== resolvers ===============

async def resolve_categories(db: AsyncSession, tenant_id: int, entities: Dict[str, Any]) -> List[str]:
    """
    Returns distinct categories for a tenant with robust fallbacks:
      1) Directly from products (tolerant is_active)
      2) From products joined via variants (tolerant is_active on both)
      3) Ignore is_active completely (products only)
      4) Ignore is_active completely (join via variants)
      If no category-like column exists, returns [] and logs an error.
    """
    cat_col, cat_name = _get_product_col(
        "category", "category_name", "product_category", "category_type", "type"
    )
    if cat_col is None:
        logging.error("resolve_categories: No category-like column found on Product.")
        return []

    # quick visibility
    try:
        tot_products = (await db.execute(
            select(func.count()).select_from(Product).where(Product.tenant_id == tenant_id)
        )).scalar() or 0
        logging.info(f"[resolve_categories] tenant={tenant_id} using_col={cat_name} total_products={tot_products}")
    except Exception:
        pass

    # 1) Primary: products by tenant, tolerant active
    conds: List[Any] = [Product.tenant_id == tenant_id]
    _add_active_trueish(Product, conds)

    cat = func.trim(cat_col).label("cat")
    stmt = (
        select(cat)
        .where(and_(*conds))
        .where(_norm_nonempty(cat_col).isnot(None))
        .distinct()
        .order_by(cat)  # Safe for DISTINCT
    )
    res = await db.execute(stmt)
    rows = [r[0] for r in res.all()]
    if rows:
        return rows

    # 2) Fallback: join via variants (tolerant active on both)
    vconds: List[Any] = [Product.id == ProductVariant.product_id, Product.tenant_id == tenant_id]
    _add_active_trueish(Product, vconds)
    _add_active_trueish(ProductVariant, vconds)

    vcat = func.trim(cat_col).label("cat")
    vstmt = (
        select(vcat)
        .join(ProductVariant, Product.id == ProductVariant.product_id)
        .where(and_(*vconds))
        .where(_norm_nonempty(cat_col).isnot(None))
        .distinct()
        .order_by(vcat)
    )
    vres = await db.execute(vstmt)
    rows2 = [r[0] for r in vres.all()]
    if rows2:
        return rows2

    # 3) Last resort: IGNORE is_active completely (products only)
    cat_any = func.trim(cat_col).label("cat")
    stmt_any = (
        select(cat_any)
        .where(Product.tenant_id == tenant_id)
        .where(_norm_nonempty(cat_col).isnot(None))
        .distinct()
        .order_by(cat_any)
    )
    any_res = await db.execute(stmt_any)
    rows3 = [r[0] for r in any_res.all()]
    if rows3:
        return rows3

    # 4) Last resort: IGNORE is_active completely (join via variants)
    vcat_any = func.trim(cat_col).label("cat")
    vstmt_any = (
        select(vcat_any)
        .join(ProductVariant, Product.id == ProductVariant.product_id)
        .where(Product.tenant_id == tenant_id)
        .where(_norm_nonempty(cat_col).isnot(None))
        .distinct()
        .order_by(vcat_any)
    )
    v_any_res = await db.execute(vstmt_any)
    rows4 = [r[0] for r in v_any_res.all()]
    if rows4:
        return rows4

    # nothing found
    logging.warning(f"[resolve_categories] No categories found for tenant={tenant_id} using_col={cat_name}")
    return []


async def resolve_colors(db: AsyncSession, tenant_id: int, entities: Dict[str, Any]) -> List[str]:
    return await _distinct_variant_col(
        db, ProductVariant.color, tenant_id, _filters_from_entities(entities)
    )


async def resolve_sizes(db: AsyncSession, tenant_id: int, entities: Dict[str, Any]) -> List[str]:
    return await _distinct_variant_col(
        db, ProductVariant.size, tenant_id, _filters_from_entities(entities)
    )


async def resolve_fabrics(db: AsyncSession, tenant_id: int, entities: Dict[str, Any]) -> List[str]:
    return await _distinct_variant_col(
        db, ProductVariant.fabric, tenant_id, _filters_from_entities(entities)
    )


async def resolve_price_range(db: AsyncSession, tenant_id: int, entities: Dict[str, Any]) -> List[str]:
    """
    Purchase-only price band:
      • is_rental = False
      • price IS NOT NULL
      • Prefer rows with rental_price NULL/0; else fallback to all purchase rows
    """
    return await _price_min_max(
        db,
        tenant_id,
        _filters_from_entities(entities),
        use_rental_price=False
    )


async def resolve_rental_price_range(db: AsyncSession, tenant_id: int, entities: Dict[str, Any]) -> List[str]:
    return await _price_min_max(
        db,
        tenant_id,
        _filters_from_entities(entities),
        use_rental_price=True
    )


async def resolve_rental_options(db: AsyncSession, tenant_id: int, entities: Dict[str, Any]) -> List[str]:
    # If any variant is rental under current filters -> include Rent; always include Purchase
    conds: List[Any] = [Product.id == ProductVariant.product_id, Product.tenant_id == tenant_id, ProductVariant.is_rental.is_(True)]
    _add_active_trueish(Product, conds)
    _add_active_trueish(ProductVariant, conds)
    conds.extend(_filters_from_entities(entities))

    stmt = select(func.count()).join(Product, Product.id == ProductVariant.product_id).where(and_(*conds))
    res = await db.execute(stmt)
    has_rent = (res.scalar() or 0) > 0
    return ["Rent", "Purchase"] if has_rent else ["Purchase"]


async def resolve_quantity_buckets(db: AsyncSession, tenant_id: int, entities: Dict[str, Any]) -> List[str]:
    # Buckets based on available_stock of variants that match filters
    conds: List[Any] = [Product.id == ProductVariant.product_id, Product.tenant_id == tenant_id]
    _add_active_trueish(Product, conds)
    _add_active_trueish(ProductVariant, conds)
    conds.extend(_filters_from_entities(entities))
    conds += [ProductVariant.available_stock.isnot(None), ProductVariant.available_stock > 0]

    bucket = func.case(
        (ProductVariant.available_stock >= 10, literal("10+")),
        (ProductVariant.available_stock >= 5, literal("5–9")),
        else_=literal("1–4"),
    )

    stmt = (
        select(bucket)
        .join(Product, Product.id == ProductVariant.product_id)
        .where(and_(*conds))
        .distinct()
    )
    res = await db.execute(stmt)
    buckets = [r[0] for r in res.all()]
    order = {"1–4": 0, "5–9": 1, "10+": 2}
    return [b for b in sorted(buckets, key=lambda x: order.get(x, 99))]


async def resolve_occasions(db: AsyncSession, tenant_id: int, entities: Dict[str, Any]) -> List[str]:
    if not HAS_OCCASION:
        return []
    # Distinct Occasion.name via M2M with variants, honoring filters
    v_conds: List[Any] = [Product.id == ProductVariant.product_id, Product.tenant_id == tenant_id]
    _add_active_trueish(Product, v_conds)
    _add_active_trueish(ProductVariant, v_conds)
    v_conds.extend(_filters_from_entities(entities))

    v_stmt = (
        select(ProductVariant.id)
        .join(Product, Product.id == ProductVariant.product_id)
        .where(and_(*v_conds))
    )
    v_ids = [r[0] for r in (await db.execute(v_stmt)).all()]
    if not v_ids:
        return []

    name_expr = func.trim(Occasion.name).label("val")
    stmt = (
        select(name_expr)
        .select_from(Occasion)
        .join(product_variant_occasions, product_variant_occasions.c.occasion_id == Occasion.id)
        .where(product_variant_occasions.c.variant_id.in_(v_ids))
        .where(_norm_nonempty(Occasion.name).isnot(None))
        .distinct()
        .order_by(name_expr)  # Safe with DISTINCT
    )
    res = await db.execute(stmt)
    return [r[0] for r in res.all()]


# Map NLP keys -> resolvers (all variant-aware)
RESOLVERS: Dict[str, Callable[[AsyncSession, int, Dict[str, Any]], Awaitable[List[str]]]] = {
    "category":      resolve_categories,         # from Product (+ dynamic col + fallbacks)
    "color":         resolve_colors,             # from ProductVariant
    "size":          resolve_sizes,              # from ProductVariant
    "fabric":        resolve_fabrics,            # from ProductVariant
    "price":         resolve_price_range,        # purchase price band (prefers rental_price NULL/0; fallback supported)
    "rental_price":  resolve_rental_price_range, # rental price band
    "rental":        resolve_rental_options,     # ["Rent","Purchase"] depending on data
    "quantity":      resolve_quantity_buckets,   # stock buckets
    "occasion":      resolve_occasions,          # only if you actually have it
}


async def fetch_attribute_values(
    db: AsyncSession,
    tenant_id: int,
    asked_now: List[str],
    entities: Dict[str, Any]
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key in asked_now or []:
        resolver = RESOLVERS.get(key)
        if not resolver:
            continue
        try:
            vals = await resolver(db, tenant_id, entities)
            if vals:
                out[key] = vals
        except Exception as e:
            logging.exception(f"Resolver failed for key={key}: {e}")
            out[key] = []
    return out


def format_inquiry_reply(
    values_by_attr: Dict[str, List[str]],
    ctx: Dict[str, Any] | None = None,
    language: str = "en-IN",
) -> str:
    """
    Render a natural-sentence reply in the detected language.

    values_by_attr: result from fetch_attribute_values, e.g. {"price": ["0 – 1699"]}
    ctx: optional accumulated entities (e.g., {"category": "Saree"}) to enrich phrasing
    language: BCP-47 like 'gu-IN', 'hi-IN', 'en-IN'
    """
    import re
    ctx = ctx or {}

    # ---- language helpers ----
    lr = (language or "en-IN").split("-")[0].lower()

    def _human_join(items: List[str]) -> str:
        items = [str(i).strip() for i in items if str(i).strip()]
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        # localized “and”
        conj = {"hi": "और", "gu": "અને"}.get(lr, "and")
        return f"{', '.join(items[:-1])} {conj} {items[-1]}"

    def _parse_min_max(s: str):
        if not isinstance(s, str):
            return None
        parts = re.split(r"\s*[–-]\s*", s.strip())
        if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
            return parts[0].strip(), parts[1].strip()
        return None

    def _rupee(n) -> str:
        try:
            n = int(float(n))
        except Exception:
            return str(n)
        return f"₹{n}"

    # ---- text templates ----
    T = {
        "en": {
            "have":       "We currently carry {x}.",
            "fabrics":    "Available fabrics include {x}.",
            "colors":     "Available colors include {x}.",
            "sizes":      "Available sizes include {x}.",
            "occasions":  "We carry outfits for {x} occasions.",
            "rentbuy":    "Both rental and purchase options are available.",
            "rent_only":  "Rental options are available.",
            "buy_only":   "Purchase options are available.",
            "price_mm":   "Price range is {mn}–{mx}.",
            "price_min":  "Starting price is {mn}.",
            "qty":        "Typical stock availability: {x}.",
            "none":       "I didn’t find details yet.",
        },
        "hi": {
            "have":       "हमारे पास {x} उपलब्ध हैं.",
            "fabrics":    "उपलब्ध फ़ैब्रिक: {x}.",
            "colors":     "उपलब्ध रंग: {x}.",
            "sizes":      "उपलब्ध साइज़: {x}.",
            "occasions":  "हम {x} अवसरों के लिए आउटफ़िट रखते हैं.",
            "rentbuy":    "भाड़े और खरीद — दोनों विकल्प उपलब्ध हैं.",
            "rent_only":  "सिर्फ़ भाड़े के विकल्प उपलब्ध हैं.",
            "buy_only":   "सिर्फ़ ख़रीद के विकल्प उपलब्ध हैं.",
            "price_mm":   "कीमत {mn}–{mx} के बीच है.",
            "price_min":  "शुरुआती कीमत {mn} है.",
            "qty":        "सामान्य स्टॉक उपलब्धता: {x}.",
            "none":       "अभी विवरण नहीं मिला.",
        },
        "gu": {
            "have":       "અમારી પાસે {x} ઉપલબ્ધ છે.",
            "fabrics":    "ઉપલબ્ધ ફેબ્રિક: {x}.",
            "colors":     "ઉપલબ્ધ કલર્સ: {x}.",
            "sizes":      "ઉપલબ્ધ સાઇઝ: {x}.",
            "occasions":  "અમે {x} પ્રસંગો માટે આઉટફિટ રાખીએ છીએ.",
            "rentbuy":    "ભાડે અને ખરીદી — બંને વિકલ્પ ઉપલબ્ધ છે.",
            "rent_only":  "માત્ર ભાડે વિકલ્પ ઉપલબ્ધ છે.",
            "buy_only":   "માત્ર ખરીદી વિકલ્પ ઉપલબ્ધ છે.",
            "price_mm":   "કિંમત {mn}–{mx} છે.",
            "price_min":  "શરૂઆતની કિંમત {mn} છે.",
            "qty":        "સામાન્ય સ્ટોક ઉપલબ્ધતા: {x}.",
            "none":       "હજુ વિગતો મળ્યાં નથી.",
        },
    }.get(lr, None)

    if T is None:
        # default to English if unsupported locale
        T = T or {
            "have": "We currently carry {x}.",
            "none": "I didn’t find details yet.",
        }

    out: List[str] = []

    # Categories: prefer DB values; else use context (same as before)
    cats = values_by_attr.get("category") or []
    if not cats:
        c = ctx.get("category")
        if isinstance(c, list):
            cats = [str(x) for x in c if str(x).strip()]
        elif isinstance(c, str) and c.strip():
            cats = [c.strip()]
    if cats:
        out.append(T["have"].format(x=_human_join(cats)))

    # Fabrics / Colors / Sizes / Occasions
    fabrics = values_by_attr.get("fabric") or []
    colors  = values_by_attr.get("color") or []
    sizes   = values_by_attr.get("size") or []
    occs    = values_by_attr.get("occasion") or []
    if fabrics: out.append(T["fabrics"].format(x=_human_join(fabrics)))
    if colors:  out.append(T["colors"].format(x=_human_join(colors)))
    if sizes:   out.append(T["sizes"].format(x=_human_join(sizes)))
    if occs:    out.append(T["occasions"].format(x=_human_join(occs)))

    # Rent vs buy options (if your resolver fills "rental")
    rentbuy = values_by_attr.get("rental") or []
    if rentbuy:
        rb = {str(x).strip().lower() for x in rentbuy}
        if "rent" in rb and "purchase" in rb:
            out.append(T["rentbuy"])
        elif "rent" in rb:
            out.append(T["rent_only"])
        elif "purchase" in rb:
            out.append(T["buy_only"])

    # Price bands (purchase)
    price_bands = values_by_attr.get("price") or []
    if price_bands:
        mm = _parse_min_max(price_bands[0])
        if mm:
            mn, mx = _rupee(mm[0]), _rupee(mm[1])
            out.append(T["price_mm"].format(mn=mn, mx=mx))
        else:
            out.append(T["price_min"].format(mn=_rupee(price_bands[0])))

    # Rental price bands
    rent_bands = values_by_attr.get("rental_price") or []
    if rent_bands:
        mm = _parse_min_max(rent_bands[0])
        if mm:
            mn, mx = _rupee(mm[0]), _rupee(mm[1])
            out.append(T["price_mm"].format(mn=mn, mx=mx))
        else:
            out.append(T["price_min"].format(mn=_rupee(rent_bands[0])))

    # Quantity buckets (stock)
    qty = values_by_attr.get("quantity") or []
    if qty:
        out.append(T["qty"].format(x=_human_join(qty)))

    return " ".join(out) if out else T["none"]