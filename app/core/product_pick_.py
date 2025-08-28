import re
import logging                                # ← add this
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from app.db.session import SessionLocal

def find_assistant_image_caption_by_msg_id(messages, msg_id: str) -> str | None:
    """
    Return the caption/text of the assistant IMAGE message that has this msg_id.
    We rely on transcript entries where meta.kind == 'image' and text holds the caption.
    """
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
    """
    If the user replied to a text prompt, walk BACKWARDS from that message
    and return the nearest previous assistant image caption.
    Assumes messages are chronological.
    """
    if not messages:
        return None
    idx = None
    for i, m in enumerate(messages):
        if m.get("msg_id") == before_msg_id:
            idx = i
            break
    # Walk backwards from idx-1 to find last assistant image
    if idx is None:
        # If we didn't find the id, still try from the end
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

def _caption_first_line(text: str | None) -> str | None:
    if not text:
        return None
    t = str(text).strip()
    return t.splitlines()[0].strip() if t else None

def _extract_url_from(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r'(https?://[^\s)\]]+)', text)
    if not m:
        return None
    return m.group(1).rstrip(').,;')

def _extract_vendor_code(url: str | None) -> str | None:
    """Pull trailing -vd123456 style codes from product URLs (case-insensitive)."""
    if not url:
        return None
    m = re.search(r'-(vd\d+)\b', url, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

async def _fetch_one_or_rollback(db, sql: str, params: dict):
    """Run SELECT … LIMIT 1; rollback tx on failure so next attempts can run."""
    try:
        res = await db.execute(text(sql), params)
        return res.fetchone()
    except (ProgrammingError, DBAPIError) as e:
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
    
async def _lookup_basic_attrs_by_ids(db, product_id: int | None, variant_id: int | None):
    """
    Returns {"category": ..., "fabric": ..., "occasion": ...} or {} if not found.
    Tries several common schema patterns; safe even if columns don't exist.
    """
    # ---------- A) VARIANT-LEVEL ----------
    if variant_id:
        # A1) Try JSON columns on variants
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

        # A2) Try direct columns on variants
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

    # ---------- B) PRODUCT-LEVEL ----------
    if product_id:
        # B1) Try JSON columns on products
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

        # B2) Try direct columns on products
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

async def _lookup_product_by_url(db, url: str):
    """
    Robust lookup:
      1) Try variant by URL (common column names), using pv.id AS variant_id.
      2) Try product by URL (common column names).
      3) Try vendor code (e.g., 'vd0275944') against sku/vendor_code on both.
    Rolls back on each failed attempt so the next query isn't blocked.
    """
    # ----- 1) variant-level by URL on common column names
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

    # ----- 2) product-level by URL on common column names
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

    # ----- 3) Try vendor code (like '-vd0275944' in the URL)
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
    1) Extract URL from caption → async DB lookup for canonical name.
    2) Else fallback to caption’s first line.
    """
    url = _extract_url_from(caption)
    got = None
    if url:
        # AsyncSession context manager
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
