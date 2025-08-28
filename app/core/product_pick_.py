# app/core/product_pick_.py
# Utilities to resolve a product/variant from a caption (URL, vendor code),
# and to fetch basic attributes (category, fabric, occasion) from Postgres.
# All DB calls are async and resilient to schema differences.

import re
import logging
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from app.db.session import SessionLocal

# ────────────────────────────────────────────────────────────────────────────────
# Caption / URL helpers
# ────────────────────────────────────────────────────────────────────────────────

def _caption_first_line(text_: str | None) -> str | None:
    """Return the first non-empty line from a caption/text."""
    if not text_:
        return None
    t = str(text_).strip()
    return t.splitlines()[0].strip() if t else None


def _extract_url_from(text_: str | None) -> str | None:
    """Extract the first URL from a caption/text."""
    if not text_:
        return None
    m = re.search(r'(https?://[^\s)\]]+)', text_)
    return m.group(1).rstrip(').,;') if m else None


def _extract_vendor_code(url: str | None) -> str | None:
    """
    Extract trailing vendor code like '-vd0275944' (case-insensitive) from a product URL.
    Returns the code uppercased (e.g., 'VD0275944') or None.
    """
    if not url:
        return None
    m = re.search(r'-(vd\d+)\b', url, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


# ────────────────────────────────────────────────────────────────────────────────
# Safe DB execution helper (async)
# ────────────────────────────────────────────────────────────────────────────────

async def _fetch_one_or_rollback(db, sql: str, params: dict):
    """
    Run a SELECT … LIMIT 1; rollback tx on failure so subsequent attempts can run.
    Returns a Row or None.
    """
    try:
        res = await db.execute(text(sql), params)
        return res.fetchone()
    except (ProgrammingError, DBAPIError) as e:
        # Very important: reset aborted tx so next query can run
        try:
            await db.rollback()
        except Exception:
            pass
        logging.debug(f"Lookup failed; rolled back. sql={sql!r} params={params!r} err={e}")
        return None
    except Exception as e:
        try:
            await db.rollback()
        except Exception:
            pass
        logging.debug(f"Lookup failed (generic); rolled back. sql={sql!r} params={params!r} err={e}")
        return None


# ────────────────────────────────────────────────────────────────────────────────
# Product/Variant resolution by URL / vendor code (async)
# ────────────────────────────────────────────────────────────────────────────────

async def _lookup_product_by_url(db, url: str):
    """
    Robust lookup:
      1) Try variant by URL on common column names, using pv.id AS variant_id.
      2) Try product by URL on common column names.
      3) Try vendor code (e.g., 'vd0275944') against sku/vendor_code on both tables.
    Rolls back on each failed attempt so that the next query isn't blocked.
    Returns dict with keys: name, product_id, variant_id, product_url  OR  None.
    """
    # 1) variant-level by URL on common column names
    variant_url_cols = ["product_url", "url", "external_url", "page_url", "source_url"]
    for col in variant_url_cols:
        row = await _fetch_one_or_rollback(db, f"""
            SELECT pv.id AS variant_id, pv.product_id, p.name AS name
            FROM product_variants pv
            LEFT JOIN products p ON p.id = pv.product_id
            WHERE pv.{col} = :url
            LIMIT 1
        """, {"url": url})
        if row:
            return {
                "name": row.name,
                "product_id": row.product_id,
                "variant_id": row.variant_id,
                "product_url": url,
            }

    # 2) product-level by URL on common column names
    product_url_cols = ["product_url", "url", "external_url", "page_url", "source_url"]
    for col in product_url_cols:
        row = await _fetch_one_or_rollback(db, f"""
            SELECT p.id AS product_id, p.name
            FROM products p
            WHERE p.{col} = :url
            LIMIT 1
        """, {"url": url})
        if row:
            return {
                "name": row.name,
                "product_id": row.product_id,
                "variant_id": None,
                "product_url": url,
            }

    # 3) Fallback: vendor code (like '-vd0275944' at the end of the URL)
    code = _extract_vendor_code(url)
    if code:
        # 3a) variants by vendor code / sku
        for col in ["vendor_code", "sku", "code"]:
            row = await _fetch_one_or_rollback(db, f"""
                SELECT pv.id AS variant_id, pv.product_id, p.name AS name
                FROM product_variants pv
                LEFT JOIN products p ON p.id = pv.product_id
                WHERE UPPER(pv.{col}) = :code
                LIMIT 1
            """, {"code": code})
            if row:
                return {
                    "name": row.name,
                    "product_id": row.product_id,
                    "variant_id": row.variant_id,
                    "product_url": url,
                }

        # 3b) products by vendor code / sku
        for col in ["vendor_code", "sku", "code", "model"]:
            row = await _fetch_one_or_rollback(db, f"""
                SELECT p.id AS product_id, p.name
                FROM products p
                WHERE UPPER(p.{col}) = :code
                LIMIT 1
            """, {"code": code})
            if row:
                return {
                    "name": row.name,
                    "product_id": row.product_id,
                    "variant_id": None,
                    "product_url": url,
                }

    return None


async def resolve_product_from_caption_async(caption: str | None):
    """
    From an assistant image caption:
      - Extract URL → async DB lookup for canonical name/ids.
      - Else fall back to caption’s first line for the name.
    Returns dict(name, product_id, variant_id, product_url).
    """
    url = _extract_url_from(caption)
    got = None
    if url:
        async with SessionLocal() as db:
            try:
                got = await _lookup_product_by_url(db, url)
            except Exception as e:
                logging.exception(f"DB lookup failed for URL {url}: {e}")

    if got and got.get("name"):
        return got

    return {
        "name": _caption_first_line(caption),
        "product_id": None,
        "variant_id": None,
        "product_url": url,
    }


# ────────────────────────────────────────────────────────────────────────────────
# Attribute (category/fabric/occasion) lookup (async)
# ────────────────────────────────────────────────────────────────────────────────

async def _lookup_basic_attrs_by_ids(db, product_id: int | None, variant_id: int | None):
    """
    Returns {"category": ..., "fabric": ..., "occasion": ...} or {} if not found.
    Tries several common schema patterns; safe even if columns don't exist.
    Order: variant JSON → variant columns → product JSON → product columns.
    """
    # A) VARIANT-LEVEL
    if variant_id:
        # A1) JSON columns on variants
        for json_col in ["metadata", "meta", "attrs", "attributes"]:
            row = await _fetch_one_or_rollback(db, f"""
                SELECT
                    {json_col}->>'category' AS category,
                    {json_col}->>'fabric'   AS fabric,
                    {json_col}->>'occasion' AS occasion
                FROM product_variants pv
                WHERE pv.id = :vid
                LIMIT 1
            """, {"vid": variant_id})
            if row and (row.category or row.fabric or row.occasion):
                return {
                    "category": row.category,
                    "fabric": row.fabric,
                    "occasion": row.occasion
                }

        # A2) Direct columns on variants
        row = await _fetch_one_or_rollback(db, """
            SELECT
                pv.category AS category,
                pv.fabric   AS fabric,
                pv.occasion AS occasion
            FROM product_variants pv
            WHERE pv.id = :vid
            LIMIT 1
        """, {"vid": variant_id})
        if row and (row.category or row.fabric or row.occasion):
            return {
                "category": row.category,
                "fabric": row.fabric,
                "occasion": row.occasion
            }

    # B) PRODUCT-LEVEL
    if product_id:
        # B1) JSON columns on products
        for json_col in ["metadata", "meta", "attrs", "attributes"]:
            row = await _fetch_one_or_rollback(db, f"""
                SELECT
                    {json_col}->>'category' AS category,
                    {json_col}->>'fabric'   AS fabric,
                    {json_col}->>'occasion' AS occasion
                FROM products p
                WHERE p.id = :pid
                LIMIT 1
            """, {"pid": product_id})
            if row and (row.category or row.fabric or row.occasion):
                return {
                    "category": row.category,
                    "fabric": row.fabric,
                    "occasion": row.occasion
                }

        # B2) Direct columns on products
        row = await _fetch_one_or_rollback(db, """
            SELECT
                p.category AS category,
                p.fabric   AS fabric,
                p.occasion AS occasion
            FROM products p
            WHERE p.id = :pid
            LIMIT 1
        """, {"pid": product_id})
        if row and (row.category or row.fabric or row.occasion):
            return {
                "category": row.category,
                "fabric": row.fabric,
                "occasion": row.occasion
            }

    return {}


# ────────────────────────────────────────────────────────────────────────────────
# Heuristic text fallback using tenant allowed lists
# ────────────────────────────────────────────────────────────────────────────────

def _pick_from_allowed(text_lc: str, allowed: list[str] | None) -> str | None:
    """Pick the best (longest) token from allowed that appears in text_lc."""
    if not text_lc or not allowed:
        return None
    ordered = sorted(allowed, key=lambda x: len(x or ""), reverse=True)
    for token in ordered:
        if token and token.lower() in text_lc:
            return token
    return None


def extract_attrs_from_text(caption: str | None,
                            allowed_categories: list[str] | None,
                            allowed_fabrics: list[str] | None,
                            allowed_occasions: list[str] | None):
    """
    Heuristic fallback: scan caption to guess attrs from allowed lists.
    """
    t = (caption or "").lower()
    return {
        "category": _pick_from_allowed(t, allowed_categories),
        "fabric":   _pick_from_allowed(t, allowed_fabrics),
        "occasion": _pick_from_allowed(t, allowed_occasions),
    }


# ────────────────────────────────────────────────────────────────────────────────
# Public async API for attrs
# ────────────────────────────────────────────────────────────────────────────────

async def get_attrs_for_product_async(product_id: int | None,
                                      variant_id: int | None) -> dict:
    """
    Open async session and try variant/product patterns; return normalized dict.
    Guarantees keys: category, fabric, occasion (values may be None).
    """
    async with SessionLocal() as db:
        attrs = await _lookup_basic_attrs_by_ids(db, product_id, variant_id)
        out = {}
        for k in ("category", "fabric", "occasion"):
            v = (attrs or {}).get(k)
            if isinstance(v, str) and not v.strip():
                v = None
            out[k] = v
        return out
def find_assistant_image_caption_by_msg_id(messages, msg_id: str) -> str | None:
    for m in messages or []:
        if (
            m.get("role") == "assistant"
            and m.get("msg_id") == msg_id
            and isinstance(m.get("meta"), dict)
            and m["meta"].get("kind") == "image"
        ):
            return m.get("text") or m.get("caption")
    return None

def find_prev_assistant_image_caption(messages, before_msg_id: str) -> str | None:
    if not messages:
        return None
    idx = None
    for i, m in enumerate(messages):
        if m.get("msg_id") == before_msg_id:
            idx = i
            break
    if idx is None:
        idx = len(messages)
    for j in range(idx - 1, -1, -1):
        m = messages[j]
        if (
            m.get("role") == "assistant"
            and isinstance(m.get("meta"), dict)
            and m["meta"].get("kind") == "image"
        ):
            return m.get("text") or m.get("caption")
    return None

def _ensure_allowed_lists(local_vars: dict):
    """
    Make sure we always have the allowed lists available in this scope,
    even if the normal (non-swipe) path hasn't populated them yet.
    """
    return (
        local_vars.get("tenant_categories", []) or [],
        local_vars.get("tenant_fabric", []) or [],
        local_vars.get("tenant_color", []) or [],
        local_vars.get("tenant_occasion", []) or [],
        local_vars.get("tenant_size", []) or [],
        local_vars.get("tenant_type", []) or [],
    )

# Optional explicit exports
__all__ = [
    "resolve_product_from_caption_async",
    "get_attrs_for_product_async",
    "extract_attrs_from_text",
]
