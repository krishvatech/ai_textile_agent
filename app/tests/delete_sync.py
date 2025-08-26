#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delete Sync — delete product variant(s) by Variant ID, plus clean Pinecone.

Mirrors your update script's style:
- Reads env exactly like variant_sync.py (PINECONE_*).
- Uses get_db_connection / close_db_connection.
- Accepts multiple Variant IDs (comma/space separated).
- Optionally purges the parent product if it becomes orphaned.
- Deletes Pinecone vector id=str(variant_id) in your namespace.

Usage:
  python delete_sync.py
    → then enter: 14, 15, 21
    → choose whether to purge orphan product (y/n)
    → confirm "YES" to proceed
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from pinecone import Pinecone

from app.db.db_connection import get_db_connection, close_db_connection


# ---------- ENV / CLIENTS (same as update_sync) ----------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "textile-products")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set")

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX)


# ---------- HELPERS ----------
def _table_exists(cursor, table_name: str) -> bool:
    """
    Check if a table exists in the current search_path.
    Works for PostgreSQL via to_regclass.
    """
    try:
        cursor.execute("SELECT to_regclass(%s)", (table_name if "." in table_name else f"public.{table_name}",))
        row = cursor.fetchone()
        return bool(row and row[0])
    except Exception:
        return False


def delete_all_tables(variant_id: int, purge_orphan_product: bool = False, delete_pinecone: bool = True) -> None:
    """
    Delete a variant from:
      - product_variant_occasions (if exists)
      - product_variants
      - (optional) parent products if no other variants remain
      - Pinecone vector id=str(variant_id)

    Prints status for each step; rolls back DB on error.
    """
    conn, cursor = get_db_connection()
    try:
        # 1) Find product_id for orphan check
        cursor.execute("SELECT product_id FROM product_variants WHERE id = %s", (variant_id,))
        row = cursor.fetchone()
        if not row:
            print(f"⚠️  Variant {variant_id} not found — skipping DB delete.")
            close_db_connection(conn, cursor)
            # still attempt Pinecone cleanup (best-effort) if asked
            if delete_pinecone:
                try:
                    pinecone_index.delete(ids=[str(variant_id)], namespace=PINECONE_NAMESPACE)
                    print(f"✅ Pinecone: deleted vector id={variant_id}")
                except Exception as e:
                    print(f"⚠️ Pinecone delete failed for {variant_id}: {e}")
            return
        product_id = row[0]

        # 2) Delete mapping rows (if table present)
        if _table_exists(cursor, "product_variant_occasions"):
            cursor.execute("DELETE FROM product_variant_occasions WHERE variant_id = %s", (variant_id,))

        # Add other mapping tables here if needed, guarded by _table_exists:
        # if _table_exists(cursor, "product_variant_images"):
        #     cursor.execute("DELETE FROM product_variant_images WHERE variant_id = %s", (variant_id,))

        # 3) Delete the variant
        cursor.execute("DELETE FROM product_variants WHERE id = %s", (variant_id,))

        # 4) Purge orphan product if requested
        if purge_orphan_product:
            cursor.execute("SELECT COUNT(*) FROM product_variants WHERE product_id = %s", (product_id,))
            remaining = cursor.fetchone()[0] or 0
            if remaining == 0:
                cursor.execute("DELETE FROM products WHERE id = %s", (product_id,))
                print(f"🗑️  Purged orphan product {product_id} (no remaining variants).")

        # 5) Commit DB changes
        conn.commit()
        print(f"✅ DB: Deleted variant {variant_id}")

    except Exception as e:
        conn.rollback()
        print(f"❌ DB error while deleting variant {variant_id}: {e}")
    finally:
        close_db_connection(conn, cursor)

    # 6) Pinecone cleanup (best-effort)
    if delete_pinecone:
        try:
            pinecone_index.delete(ids=[str(variant_id)], namespace=PINECONE_NAMESPACE)
            print(f"✅ Pinecone: deleted vector id={variant_id}")
        except Exception as e:
            print(f"⚠️ Pinecone delete failed for {variant_id}: {e}")


# ---------- CLI ----------
if __name__ == "__main__":
    # Multiple variant ID input (same UX as update_sync)
    raw_input_str = input("Multiple Variant IDs dijiye (comma or space separated, e.g. 14,15,21): ").strip()
    ids = [x.strip() for x in raw_input_str.replace(",", " ").split() if x.strip().isdigit()]
    variant_ids = [int(x) for x in ids]
    if not variant_ids:
        print("❌ Koi valid variant ID nahi diya.")
        raise SystemExit(1)

    purge_ans = input("Parent product ko bhi delete karein agar orphan ho jaye? (y/n): ").strip().lower()
    purge_orphan = purge_ans in {"y", "yes", "haan", "ha", "h"}

    print(f"\nAbout to DELETE {len(variant_ids)} variant(s): {', '.join(map(str, variant_ids))}")
    print("⚠️  This is irreversible. Type YES to continue.")
    confirm = input("> ").strip()
    if confirm != "YES":
        print("Cancelled.")
        raise SystemExit(0)

    for vid in variant_ids:
        print(f"\n— Deleting Variant ID: {vid} —")
        delete_all_tables(vid, purge_orphan_product=purge_orphan, delete_pinecone=True)
