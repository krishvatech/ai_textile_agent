# app/services/product_status_sync.py
from __future__ import annotations

import os
from typing import List, Optional, Dict, Any, Sequence

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from fastapi import HTTPException

from dotenv import load_dotenv
from pinecone import Pinecone

# -------------------------------------------------------------------
# Pinecone setup (single init used for ALL ops: update + delete)
# -------------------------------------------------------------------
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


# =========================
# Internal helpers (status)
# =========================
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


# ===================
# Public API (status)
# ===================
async def toggle_product_active(
    db: AsyncSession,
    tenant_id: int,
    product_id: int,
    new_state: Optional[bool] = None,
) -> Dict[str, object]:
    """
    Flip all variants of a product to active/deactive in Postgres,
    then mirror `is_active` in Pinecone.
    """
    if not await _product_belongs_to_tenant(db, product_id, tenant_id):
        raise ValueError("Product not found for this tenant.")

    current = await _any_variant_active(db, product_id)
    state = (not current) if new_state is None else bool(new_state)

    ids = await _set_all_variants_state(db, product_id, state)
    await db.commit()

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


# check table existence safely
async def _table_exists(db: AsyncSession, table_name: str) -> bool:
    q = await db.execute(
        text("SELECT to_regclass(:tname) IS NOT NULL"),
        {"tname": f"public.{table_name}" if "." not in table_name else table_name},
    )
    return bool(q.scalar())

def _chunk(lst: Sequence, n: int = 1000):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# delete from known mapping tables that reference variant_id
async def _delete_variant_dependents(db: AsyncSession, variant_ids: list[int]) -> None:
    if not variant_ids:
        return

    # list any mapping tables you use that have a "variant_id" FK
    candidate_tables = [
        "product_variant_occasions",
        "product_variant_images",
        "product_variant_sizes",
        "product_variant_colors",
        "product_variant_attributes",
    ]

    # filter to only existing tables
    existing = []
    for tbl in candidate_tables:
        if await _table_exists(db, tbl):
            existing.append(tbl)

    # delete in chunks for each existing table
    for tbl in existing:
        for batch in _chunk(variant_ids, 1000):
            # ANY(:ids) works with asyncpg as an int[] bind
            await db.execute(
                text(f"DELETE FROM {tbl} WHERE variant_id = ANY(:ids)"),
                {"ids": batch},
            )


async def delete_variant_everywhere(
    db: AsyncSession,
    tenant_id: int,
    product_id: int,
    variant_id: int,
) -> dict:
    # Ownership + existence checks
    ok = await db.execute(
        text("SELECT 1 FROM products WHERE id=:pid AND tenant_id=:tid"),
        {"pid": product_id, "tid": tenant_id},
    )
    if not ok.first():
        raise HTTPException(status_code=404, detail="Product not found")

    vok = await db.execute(
        text("SELECT 1 FROM product_variants WHERE id=:vid AND product_id=:pid"),
        {"vid": variant_id, "pid": product_id},
    )
    if not vok.first():
        raise HTTPException(status_code=404, detail="Variant not found")

    # Pinecone first (fail-fast)
    try:
        _pinecone_delete_variants_by_ids([variant_id])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pinecone delete failed: {e}")

    # DB: dependents -> variant
    try:
        # remove rows in mapping tables that reference variant_id
        await _delete_variant_dependents(db, [variant_id])

        # finally remove the variant
        await db.execute(
            text("DELETE FROM product_variants WHERE id=:vid AND product_id=:pid"),
            {"vid": variant_id, "pid": product_id},
        )

        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"DB delete failed: {e}")

    return {
        "ok": True,
        "action": "deleted_variant",
        "product_id": product_id,
        "deleted_variant_id": variant_id,
        "pinecone_index": PINECONE_INDEX,
        "pinecone_namespace": PINECONE_NAMESPACE,
    }

# ==========================================
# Pinecone metadata sync for a single variant
# ==========================================
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
    if not _index:
        return

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


# ================================
# Bulk push product-level metadata
# ================================
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

async def push_product_metadata_to_pinecone(
    db: AsyncSession,
    product_id: int,
    *,
    product: str | None = None,
    category: str | None = None,
    description: str | None = None,
    product_url: str | None = None,
) -> dict:
    vids = await _get_variant_ids_by_product(db, product_id)
    if not vids:
        return {"product_id": product_id, "variant_count": 0, "pushed": False}

    meta = {
        "name": product,
        "category": category,
        "description": description,
        "product_url": product_url,
    }
    _bulk_update_pinecone_metadata(vids, meta)
    return {"product_id": product_id, "variant_count": len(vids), "pushed": True}


# ======================
# Delete (DB + Pinecone)
# ======================
def _chunk(lst: Sequence, n: int = 1000):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

async def _ensure_product_belongs_to_tenant_or_404(
    db: AsyncSession, product_id: int, tenant_id: int
) -> None:
    q = await db.execute(
        text("SELECT 1 FROM products WHERE id = :pid AND tenant_id = :tid"),
        {"pid": product_id, "tid": tenant_id},
    )
    if not q.first():
        raise HTTPException(status_code=404, detail="Product not found")

async def _collect_variant_ids(db: AsyncSession, product_id: int, tenant_id: int) -> List[int]:
    # Use the correct table AND tenant guard via join
    q = await db.execute(
        text("""
            SELECT pv.id
            FROM product_variants pv
            JOIN products p ON p.id = pv.product_id
            WHERE pv.product_id = :pid AND p.tenant_id = :tid
        """),
        {"pid": product_id, "tid": tenant_id},
    )
    return [r[0] for r in q.fetchall()]

def _pinecone_delete_variants_by_ids(variant_ids: List[int]) -> None:
    if not _index or not variant_ids:
        return
    ids = [str(vid) for vid in variant_ids]   # change to f"variant:{vid}" if you used a prefix
    for batch in _chunk(ids, 1000):
        try:
            _index.delete(ids=batch, namespace=PINECONE_NAMESPACE)
        except Exception as e:
            print(f"[pinecone] delete batch failed ({len(batch)} ids): {e}")
            raise

async def delete_product_everywhere(
    db: AsyncSession, tenant_id: int, product_id: int
) -> Dict[str, Any]:
    """
    1) Verify ownership
    2) Collect connected variant IDs
    3) Delete Pinecone vectors (fail-fast)
    4) Delete DB rows: product_variants then products (transactional)
    """
    await _ensure_product_belongs_to_tenant_or_404(db, product_id, tenant_id)

    variant_ids = await _collect_variant_ids(db, product_id, tenant_id)

    # Pinecone first (fail-fast)
    try:
        _pinecone_delete_variants_by_ids(variant_ids)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pinecone delete failed: {e}")

    # DB delete
    try:

        # remove dependent rows that reference variant_id (fixes your FK error)
        await _delete_variant_dependents(db, variant_ids)

        # Variants
        await db.execute(
            text("DELETE FROM product_variants WHERE product_id = :pid"),
            {"pid": product_id},
        )
        # Product
        await db.execute(
            text("DELETE FROM products WHERE id = :pid AND tenant_id = :tid"),
            {"pid": product_id, "tid": tenant_id},
        )
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"DB delete failed: {e}")

    return {
        "ok": True,
        "action": "deleted_product_and_variants",
        "deleted_product_id": product_id,
        "deleted_variant_ids": variant_ids,
        "pinecone_index": PINECONE_INDEX,
        "pinecone_namespace": PINECONE_NAMESPACE,
    }
