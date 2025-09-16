# app/services/product_status_sync.py
from __future__ import annotations

import os
from typing import List, Optional, Dict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from dotenv import load_dotenv
from pinecone import Pinecone

# --- Pinecone setup (same style as variant_sync.py) --------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "textile-products")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

_pine = None
_index = None
if PINECONE_API_KEY:
    try:
        _pine = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pine.Index(PINECONE_INDEX)
    except Exception as e:
        print(f"[pinecone] init failed: {e}")
        _index = None


# --- Internal helpers --------------------------------------------------
async def _product_belongs_to_tenant(db: AsyncSession, product_id: int, tenant_id: int) -> bool:
    r = await db.execute(
        text("SELECT 1 FROM products WHERE id=:pid AND tenant_id=:tid LIMIT 1"),
        {"pid": product_id, "tid": tenant_id},
    )
    return bool(r.first())

async def _any_variant_active(db: AsyncSession, product_id: int) -> bool:
    r = await db.execute(
        text("SELECT COALESCE(BOOL_OR(is_active), FALSE) FROM product_variants WHERE product_id=:pid"),
        {"pid": product_id},
    )
    row = r.first()
    return bool(row[0]) if row else False

async def _set_all_variants_state(db: AsyncSession, product_id: int, is_active: bool) -> List[int]:
    # Update state
    await db.execute(
        text("""UPDATE product_variants
                SET is_active=:state, updated_at=NOW()
                WHERE product_id=:pid"""),
        {"state": is_active, "pid": product_id},
    )
    # Return updated variant ids
    res = await db.execute(
        text("SELECT id FROM product_variants WHERE product_id=:pid"),
        {"pid": product_id},
    )
    return [row[0] for row in res.fetchall()]

def _push_is_active_to_pinecone(variant_ids: List[int], is_active: bool) -> None:
    if not _index or not variant_ids:
        return
    for vid in variant_ids:
        try:
            _index.update(
                id=str(vid),
                set_metadata={"is_active": bool(is_active)},
                namespace=PINECONE_NAMESPACE,
            )
        except Exception as e:
            print(f"[pinecone] update failed for {vid}: {e}")


# --- Public API --------------------------------------------------------
async def toggle_product_active(
    db: AsyncSession,
    tenant_id: int,
    product_id: int,
    new_state: Optional[bool] = None,
) -> Dict[str, object]:
    """
    Flip all variants of a product to active/deactive in Postgres,
    then mirror the same `is_active` metadata in Pinecone.
    If `new_state` is None, it toggles (based on any variant being active).
    """
    if not await _product_belongs_to_tenant(db, product_id, tenant_id):
        raise ValueError("Product not found for this tenant.")

    current = await _any_variant_active(db, product_id)
    state = (not current) if new_state is None else bool(new_state)

    ids = await _set_all_variants_state(db, product_id, state)
    await db.commit()

    # Pinecone mirror (best-effort)
    try:
        _push_is_active_to_pinecone(ids, state)
    except Exception:
        pass

    return {
        "product_id": product_id,
        "variant_ids": ids,
        "new_state": state,
        "variant_count": len(ids),
    }


async def set_variant_active(
    db: AsyncSession,
    tenant_id: int,
    variant_id: int,
    is_active: bool,
) -> Dict[str, object]:
    """
    Set a single variant's active state (with tenant guard) and mirror to Pinecone.
    """
    res = await db.execute(
        text("""SELECT pv.id, pv.product_id
                FROM product_variants pv
                JOIN products p ON p.id = pv.product_id
                WHERE pv.id=:vid AND p.tenant_id=:tid"""),
        {"vid": variant_id, "tid": tenant_id},
    )
    row = res.first()
    if not row:
        raise ValueError("Variant not found for this tenant.")

    await db.execute(
        text("UPDATE product_variants SET is_active=:state, updated_at=NOW() WHERE id=:vid"),
        {"state": is_active, "vid": variant_id},
    )
    await db.commit()

    try:
        _push_is_active_to_pinecone([variant_id], is_active)
    except Exception:
        pass

    return {"variant_id": variant_id, "product_id": row[1], "new_state": bool(is_active)}


# --- Pinecone metadata sync for a single variant (edit) ----------------

def push_variant_metadata_to_pinecone(
    variant_id: int,
    *,
    color=None,
    size=None,
    fabric=None,
    price=None,
    rental_price=None,
    available_stock=None,
    is_rental=None,
    image_url=None,
    is_active=None,
    product_url=None,
) -> None:
    """
    Update Pinecone metadata for one variant id. Call this after
    you UPDATE the variant row in Postgres.
    """
    if not _index:
        return

    # keep only keys that have a non-None value so we don't overwrite with nulls
    meta = {
        "color": color,
        "size": size,
        "fabric": fabric,
        "price": price,
        "rental_price": rental_price,
        "available_stock": available_stock,
        "is_rental": bool(is_rental) if is_rental is not None else None,
        "image_url": image_url,
        "is_active": bool(is_active) if is_active is not None else None,
        "product_url": product_url,
    }
    clean = {k: v for k, v in meta.items() if v is not None}

    if not clean:
        return

    try:
        _index.update(
            id=str(variant_id),
            set_metadata=clean,
            namespace=PINECONE_NAMESPACE,
        )
    except Exception as e:
        print(f"[pinecone] variant {variant_id} metadata update failed: {e}")


# ---- Bulk Pinecone metadata helpers (generic) -------------------------

from typing import Dict, List

async def _get_variant_ids_by_product(db: AsyncSession, product_id: int) -> List[int]:
    res = await db.execute(
        text("SELECT id FROM product_variants WHERE product_id = :pid"),
        {"pid": product_id},
    )
    return [r[0] for r in res.fetchall()]

def _bulk_update_pinecone_metadata(variant_ids: List[int], metadata: Dict):
    if not _index or not variant_ids or not metadata:
        return
    clean = {k: v for k, v in metadata.items() if v is not None}
    if not clean:
        return
    for vid in variant_ids:
        try:
            _index.update(
                id=str(vid),
                set_metadata=clean,
                namespace=PINECONE_NAMESPACE,
            )
        except Exception as e:
            print(f"[pinecone] bulk meta update failed for {vid}: {e}")

# ---- Product → all variants: push product fields into Pinecone --------

async def push_product_metadata_to_pinecone(
    db: AsyncSession,
    product_id: int,
    *,
    product: str | None = None,
    category: str | None = None,
    description: str | None = None,
    product_url: str | None = None,
) -> dict:
    """
    Push product-level fields to *all* variant vectors of this product.
    Call this after updating products table.
    """
    vids = await _get_variant_ids_by_product(db, product_id)
    if not vids:
        return {"product_id": product_id, "variant_count": 0, "pushed": False}

    # Use keys you want visible in Pinecone metadata
    meta = {
        "name": product,                       # product name
        "category": category,
        "description": description,
        "product_url": product_url,
    }
    _bulk_update_pinecone_metadata(vids, meta)
    return {"product_id": product_id, "variant_count": len(vids), "pushed": True}
