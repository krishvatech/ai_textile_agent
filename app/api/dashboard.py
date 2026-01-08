# app/api/dashboard.py
from __future__ import annotations

import os
import json
import re
from io import BytesIO
from urllib.parse import urlparse, quote_plus
from typing import List, Dict, Any, Optional
from fastapi import UploadFile, File, status

import httpx
from fastapi import APIRouter, Request, HTTPException, Depends, Query, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse,JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
# top of file (with other imports)
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.product_status_sync import toggle_product_active,push_variant_metadata_to_pinecone,push_product_metadata_to_pinecone,delete_product_everywhere,delete_variant_everywhere,upsert_variant_image_from_db
from app.services.product_modelling.service import generate_catalog, load_model_library_index, MAX_IMAGE_BYTES

# Use the same get_db that admin.py relies on
from app.db.session import get_db  # <-- same pattern as admin.py

router = APIRouter()
templates = Jinja2Templates(directory="app/templates/tenants")

# ---------------------------------------------------------------------
# Auth: only tenant_admins can use the tenant dashboard
# ---------------------------------------------------------------------
ALLOWED_DASHBOARD_ROLES = {"tenant_admin"}

def require_auth(request: Request):
    role = request.session.get("role")
    tid = request.session.get("tenant_id")
    if (not role) or (role not in ALLOWED_DASHBOARD_ROLES) or (tid is None):
        nxt = request.url.path or "/dashboard"
        return RedirectResponse(url=f"/login?next={nxt}", status_code=303)
    return int(tid)

async def get_tenant_name(tenant_id: int, db: AsyncSession) -> str:
    res = await db.execute(
        text("SELECT name FROM tenants WHERE id = :id LIMIT 1"),
        {"id": tenant_id},
    )
    row = res.fetchone()
    if not row:
        return "Unknown"
    name = row[0]
    return (name.strip() if isinstance(name, str) else name) or "Unknown"

# ---------------------------------------------------------------------
# Meta / WhatsApp media proxy (server-side fetch using your token)
# ---------------------------------------------------------------------
ALLOWED_WA_HOSTS = {
    "lookaside.fbsbx.com",
    "graph.facebook.com",
    "scontent.xx.fbcdn.net",
    "scontent.cdninstagram.com",
}
MID_RE = re.compile(r"[?&]mid=([0-9A-Za-z_-]+)")

def get_waba_token_for_tenant(tenant_id: int) -> Optional[str]:
    """
    Prefer per-tenant secret; fall back to a global one.
    .env examples:
      WABA_TOKEN=EAAG...           (global)
      WABA_TOKEN_4=EAAG...         (tenant_id=4)
    """
    return (
        os.getenv("WHATSAPP_TOKEN")
    )

@router.get("/modelling", response_class=HTMLResponse)
async def tenant_modelling_page(request: Request, db: AsyncSession = Depends(get_db)):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard

    tenant_id = guard
    tenant_name = request.session.get("tenant_name") or await get_tenant_name(tenant_id, db)

    return templates.TemplateResponse(
        "product_modelling.html",
        {"request": request, "tenant_id": tenant_id, "tenant_name": tenant_name},
    )


@router.post("/modelling/generate", response_class=JSONResponse)
async def tenant_modelling_generate(
    request: Request,
    product_image: UploadFile = File(None),
    image: UploadFile = File(None),  # backward-compat
    reference_image: UploadFile = File(None),
    model_image: UploadFile = File(None),
    workflow: str = Form("auto"),
    predefined_model_id: str = Form(""),
    style_preset: str = Form(""),
    num_images: int = Form(4),
    background: str = Form("white"),
    pose_set: str = Form("ecom6"),
    strict_garment: str = Form("1"),
):
    # JSON auth (no redirect)
    role = request.session.get("role")
    tenant_id = request.session.get("tenant_id")
    if role != "tenant_admin" or tenant_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authorized")

    image = product_image or image
    if not image:
        raise HTTPException(status_code=400, detail="Product image is required.")
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image.")

    data = await image.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Product image is too large.")

    reference_bytes = None
    if reference_image and reference_image.filename:
        if not reference_image.content_type or not reference_image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Reference image must be an image.")
        reference_bytes = await reference_image.read()
        if len(reference_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Reference image is too large.")

    model_bytes = None
    if model_image and model_image.filename:
        if not model_image.content_type or not model_image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Model image must be an image.")
        model_bytes = await model_image.read()
        if len(model_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Model image is too large.")

    try:
        result = await generate_catalog(
            workflow=workflow,
            product_image_bytes=data,
            reference_image_bytes=reference_bytes,
            model_image_bytes=model_bytes,
            predefined_model_id=(predefined_model_id or "").strip() or None,
            style_preset=(style_preset or "").strip() or None,
            background=background,
            pose_set=pose_set,
            strict_garment=(str(strict_garment) == "1"),
            num_images=num_images,
            tenant_scope=f"tenant_{tenant_id}",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vertex AI failed: {e}")

    return JSONResponse(result)


@router.get("/modelling/models", response_class=JSONResponse)
async def tenant_modelling_models(request: Request):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    return JSONResponse(load_model_library_index())

@router.get("/media/wa/{media_id}")
async def wa_media_by_id(request: Request, media_id: str):
    """Stream a WhatsApp media file by media_id using the Graph API."""
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    token = get_waba_token_for_tenant(tenant_id)
    if not token:
        return HTMLResponse("WABA token missing on server", status_code=500)

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        # 1) Lookup the media URL
        meta = await client.get(
            f"https://graph.facebook.com/v20.0/{media_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if meta.status_code >= 400:
            return HTMLResponse(f"Meta lookup failed: {meta.text}", status_code=meta.status_code)
        info = meta.json() or {}
        url = info.get("url")
        if not url:
            return HTMLResponse("Media URL missing", status_code=502)

        # 2) Download using the same auth
        file_res = await client.get(url, headers={"Authorization": f"Bearer {token}"})
        if file_res.status_code >= 400:
            return HTMLResponse(f"Media download failed: {file_res.text}", status_code=file_res.status_code)

        return StreamingResponse(BytesIO(file_res.content),
                                 media_type=file_res.headers.get("Content-Type", "application/octet-stream"))

@router.get("/media/wa")
async def wa_media_by_url(request: Request, url: str = Query(...)):
    """Stream a WhatsApp media file by a full (lookaside/graph) URL."""
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    token = get_waba_token_for_tenant(tenant_id)
    if not token:
        return HTMLResponse("WABA token missing on server", status_code=500)

    parsed = urlparse(url)
    if parsed.hostname not in ALLOWED_WA_HOSTS:
        return HTMLResponse("Host not allowed", status_code=400)

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        r = await client.get(url, headers={"Authorization": f"Bearer {token}"})
        if r.status_code >= 400:
            return HTMLResponse(f"Download failed: {r.text}", status_code=r.status_code)
        return StreamingResponse(BytesIO(r.content),
                                 media_type=r.headers.get("Content-Type", "application/octet-stream"))

# ---------------------------------------------------------------------
# Dashboard home
# ---------------------------------------------------------------------
@router.get("/", response_class=HTMLResponse, name="dashboard_home")
async def dashboard_home(request: Request, db: AsyncSession = Depends(get_db)):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    # --- KPIs ---
    res = await db.execute(text("SELECT COUNT(*) FROM customers WHERE tenant_id = :tid"), {"tid": tenant_id})
    customers_count = res.scalar() or 0

    res = await db.execute(text("SELECT COUNT(*) FROM products WHERE tenant_id = :tid"), {"tid": tenant_id})
    products_count = res.scalar() or 0

    res = await db.execute(text("""
        SELECT COUNT(DISTINCT p.id)
        FROM products p
        JOIN product_variants pv ON pv.product_id = p.id
        WHERE p.tenant_id = :tid AND COALESCE(pv.is_rental,false) = true
    """), {"tid": tenant_id})
    rental_products_count = res.scalar() or 0

    res = await db.execute(text("""
        SELECT COUNT(DISTINCT p.id)
        FROM products p
        JOIN product_variants pv ON pv.product_id = p.id
        WHERE p.tenant_id = :tid AND COALESCE(pv.is_rental,false) = false
    """), {"tid": tenant_id})
    selling_products_count = res.scalar() or 0

    res = await db.execute(text("SELECT COUNT(*) FROM orders WHERE tenant_id = :tid"), {"tid": tenant_id})
    orders_count = res.scalar() or 0

    res = await db.execute(text("SELECT COUNT(*) FROM rentals WHERE tenant_id = :tid"), {"tid": tenant_id})
    rentals_count = res.scalar() or 0

    # --- Charts ---
    res = await db.execute(text("""
        SELECT DATE_TRUNC('month', start_date) AS mth, COUNT(*)
        FROM orders
        WHERE tenant_id = :tid
        GROUP BY mth
        ORDER BY mth
    """), {"tid": tenant_id})
    sell_rows = res.fetchall() or []

    res = await db.execute(text("""
        SELECT DATE_TRUNC('month', rental_start_date) AS mth, COUNT(*)
        FROM rentals
        WHERE tenant_id = :tid
        GROUP BY mth
        ORDER BY mth
    """), {"tid": tenant_id})
    rent_rows = res.fetchall() or []

    # Merge months → labels + two datasets
    from collections import OrderedDict
    merged = OrderedDict()
    for m, c in sell_rows:
        merged[m] = {"sell": c, "rent": 0}
    for m, c in rent_rows:
        merged.setdefault(m, {"sell": 0, "rent": 0})
        merged[m]["rent"] += c

    labels = [m.strftime("%b %Y") for m in merged.keys()]
    sell_series = [v["sell"] for v in merged.values()]
    rent_series = [v["rent"] for v in merged.values()]

    tenant_name = request.session.get("tenant_name") or await get_tenant_name(tenant_id, db)

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "customers_count": customers_count,
            "products_count": products_count,
            "rental_products_count": rental_products_count,
            "selling_products_count": selling_products_count,
            "orders_count": orders_count,
            "rentals_count": rentals_count,
            "chart_labels": labels,
            "chart_sell": sell_series,
            "chart_rent": rent_series,
        },
    )

# ---------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------

# -----------Delete Customer ---------------------
@router.post("/customer/{customer_id}/delete")
async def customer_delete(request: Request, customer_id: int, db: AsyncSession = Depends(get_db)):
    # Only tenant_admins reach here because dashboard already gates on role
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    # 1) Check that the customer belongs to this tenant
    res = await db.execute(
        text("SELECT id FROM customers WHERE id = :cid AND tenant_id = :tid LIMIT 1"),
        {"cid": customer_id, "tid": tenant_id},
    )
    row = res.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Customer not found")

    # 2) Delete dependent rows FIRST, then the customer
    #    (keep this list in sync with your real schema / FKs)
    try:
        # Chat sessions
        await db.execute(
            text("DELETE FROM chat_sessions WHERE customer_id = :cid"),
            {"cid": customer_id},
        )

        # Orders
        await db.execute(
            text("DELETE FROM orders WHERE tenant_id = :tid AND customer_id = :cid"),
            {"tid": tenant_id, "cid": customer_id},
        )

        # Rentals
        await db.execute(
            text("DELETE FROM rentals WHERE tenant_id = :tid AND customer_id = :cid"),
            {"tid": tenant_id, "cid": customer_id},
        )

        # Finally, the customer row
        await db.execute(
            text("DELETE FROM customers WHERE id = :cid AND tenant_id = :tid"),
            {"cid": customer_id, "tid": tenant_id},
        )

        await db.commit()
    except Exception as e:
        # optional: log e
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete customer")

    # 3) Back to list with a tiny flag for UI flash
    return RedirectResponse(url="/dashboard/customer?deleted=1", status_code=303)


@router.get("/customer", response_class=HTMLResponse)
async def customers_list(request: Request, db: AsyncSession = Depends(get_db)):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    res = await db.execute(text("""
        SELECT id, whatsapp_id, name, preferred_language, phone, email, created_at, is_active, loyalty_points
        FROM customers
        WHERE tenant_id = :tid
        ORDER BY created_at DESC
        LIMIT 200
    """), {"tid": tenant_id})
    rows = res.fetchall() or []

    customers = [
        {
            "id": r[0],
            "whatsapp_id": r[1] or "-",
            "name": (r[2] or "(no name)"),
            "preferred_language": r[3] or "—",
            "phone": r[4] or "—",
            "email": r[5] or "—",
            "created_at": r[6],
            "is_active": bool(r[7]) if r[7] is not None else False,
            "loyalty_points": r[8] or 0,
        }
        for r in rows
    ]
    tenant_name = request.session.get("tenant_name") or await get_tenant_name(tenant_id, db)

    return templates.TemplateResponse(
        "customer_list.html",
        {"request": request, "tenant_id": tenant_id, "tenant_name": tenant_name, "customers": customers},
    )

@router.get("/customer/{customer_id}", response_class=HTMLResponse)
async def customer_detail(request: Request, customer_id: int, db: AsyncSession = Depends(get_db)):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    # ---- Customer row
    res = await db.execute(text("""
        SELECT id, name, phone, email, preferred_language, loyalty_points, created_at
        FROM customers
        WHERE id = :cid AND tenant_id = :tid
        LIMIT 1
    """), {"cid": customer_id, "tid": tenant_id})
    row = res.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Customer not found")

    customer = {
        "id": row[0],
        "name": row[1] or "(no name)",
        "phone": row[2] or "-",
        "email": row[3] or "-",
        "preferred_language": row[4] or "-",
        "loyalty_points": row[5] or 0,
        "created_at": row[6],
    }

    # ---- Recent Orders (last 10)
    res = await db.execute(text("""
        SELECT o.id, o.order_type, o.status, o.price, o.created_at,
               pv.color, pv.size, pv.fabric, p.name
        FROM orders o
        JOIN product_variants pv ON pv.id = o.product_variant_id
        JOIN products p         ON p.id = pv.product_id
        WHERE o.customer_id = :cid AND o.tenant_id = :tid
        ORDER BY o.created_at DESC
        LIMIT 10
    """), {"cid": customer_id, "tid": tenant_id})
    orders_rows = res.fetchall() or []
    orders = [{
        "id": r[0],
        "order_type": r[1],
        "status": r[2],
        "price": r[3],
        "created_at": r[4],
        "variant": f"{r[8]} — {r[5]}/{r[6]}/{r[7]}",
    } for r in orders_rows]

    # ---- Recent Rentals (last 10)
    res = await db.execute(text("""
        SELECT r.id, r.status, r.rental_price, r.rental_start_date, r.rental_end_date, r.created_at,
               pv.color, pv.size, pv.fabric, p.name
        FROM rentals r
        JOIN product_variants pv ON pv.id = r.product_variant_id
        JOIN products p         ON p.id = pv.product_id
        WHERE r.customer_id = :cid AND r.tenant_id = :tid
        ORDER BY r.created_at DESC
        LIMIT 10
    """), {"cid": customer_id, "tid": tenant_id})
    rentals_rows = res.fetchall() or []
    rentals = [{
        "id": r[0],
        "status": r[1],
        "rental_price": r[2],
        "start_date": r[3],
        "end_date": r[4],
        "created_at": r[5],
        "variant": f"{r[9]} — {r[6]}/{r[7]}/{r[8]}",
    } for r in rentals_rows]

    tenant_name = request.session.get("tenant_name") or await get_tenant_name(tenant_id, db)
    return templates.TemplateResponse(
        "customer_detail.html",
        {
            "request": request,
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "customer": customer,
            "orders": orders,
            "rentals": rentals,
        },
    )

# ---------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------
# --------------------------------------------------------------------
# PRoducts add By Tenants-Admin
# --------------------------------------------------------------------

@router.post("/product/{product_id}/toggle-active")
async def product_toggle_active(request: Request, product_id: int, db: AsyncSession = Depends(get_db)):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    try:
        await toggle_product_active(db, tenant_id, product_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")

    return RedirectResponse(url="/dashboard/product?status_toggled=1", status_code=303)


@router.get("/product/add", response_class=HTMLResponse)
async def product_add_form(request: Request, db: AsyncSession = Depends(get_db)):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard

    res = await db.execute(text("SELECT id, name FROM occasions ORDER BY name ASC"))
    occasions = [{"id": r[0], "name": r[1]} for r in res.fetchall()]

    return templates.TemplateResponse(
        "product_add.html",
        {"request": request, "form_error": None, "occasions": occasions},
    )


# --- PRODUCT: Add (POST) -----------------------------
@router.post("/product/add")
async def product_add_submit(
    request: Request,
    db: AsyncSession = Depends(get_db),

    # fields from product_add.html
    product: str = Form(...),            # -> products.name
    category: str = Form(""),
    description: str = Form(""),
    image_url: str = Form(""),
    product_url: str = Form(""),
    type: str = Form("Women"),           # Men / Women

    # initial variant (optional)
    price: float | None = Form(None),
    rental_price: float | None = Form(None),
    fabric: str = Form(""),
    color: str = Form(""),
    size: str = Form(""),
    stock: int | None = Form(None),
    sellable: bool = Form(False),
    rentable: bool = Form(False),

    # occasion (single select)
    occasion_id: str = Form(""),
):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    name_clean = (product or "").strip()
    if not name_clean:
        return templates.TemplateResponse(
            "product_add.html",
            {"request": request, "form_error": "Product name is required."},
            status_code=400,
        )

    try:
        # 1) Create product (store type/product_url + timestamps)
        r = await db.execute(
            text("""
                INSERT INTO products
                    (tenant_id, name, type, category, description, product_url, created_at, updated_at)
                VALUES
                    (:tid, :name, :type, :cat, :desc, :purl, NOW(), NOW())
                RETURNING id
            """),
            {
                "tid": tenant_id,
                "name": name_clean,
                "type": (type or "").strip() or None,
                "cat": (category or "").strip() or None,
                "desc": (description or "").strip() or None,
                "purl": (product_url or "").strip() or None,
            },
        )
        product_id = r.scalar_one()

        # 2) Create initial variant(s) and collect IDs
        variant_ids: list[int] = []

        base = {
            "product_id": product_id,
            "color": (color or None),
            "size": (size or None),
            "fabric": (fabric or None),
            "price": price,
            "rental_price": rental_price,
            "available_stock": stock or 0,
            "image_url": (image_url or None),
            "is_active": True,
            "product_url": (product_url or None),
        }

        async def _insert_variant(is_rental: bool) -> int:
            res = await db.execute(
                text("""
                    INSERT INTO product_variants
                        (product_id, color, size, fabric, price, rental_price, available_stock,
                         is_rental, image_url, is_active, product_url, created_at, updated_at)
                    VALUES
                        (:product_id, :color, :size, :fabric, :price, :rental_price, :available_stock,
                         :is_rental, :image_url, :is_active, :product_url, NOW(), NOW())
                    RETURNING id
                """),
                {**base, "is_rental": is_rental},
            )
            return res.scalar_one()

        # If neither is checked, default to SELL
        if sellable or (not sellable and not rentable):
            variant_ids.append(await _insert_variant(False))
        if rentable:
            variant_ids.append(await _insert_variant(True))

        # 3) Map selected occasion to each variant (if any)
        if occasion_id and occasion_id.isdigit():
            oid = int(occasion_id)
            for vid in variant_ids:
                await db.execute(
                    text("""
                        INSERT INTO product_variant_occasions (variant_id, occasion_id)
                        VALUES (:vid, :oid)
                    """),
                    {"vid": vid, "oid": oid},
                )

        # 4) Commit DB so Pinecone readers see fresh rows
        await db.commit()

    except Exception:
        await db.rollback()
        return templates.TemplateResponse(
            "product_add.html",
            {"request": request, "form_error": "Failed to save product. Please check fields and try again."},
            status_code=400,
        )

    # 5) Pinecone upserts: CLIP embed from image_url + full metadata (incl. type/occasion)
    try:
        for vid in variant_ids:
            await upsert_variant_image_from_db(db, vid)

        await push_product_metadata_to_pinecone(
            db,
            product_id,
            product=name_clean,
            category=(category or None),
            description=(description or None),
            product_url=((product_url or "").strip() or None),
            # include type if your helper accepts it:
            # type=(type or None),
        )
    except Exception:
        pass

    return RedirectResponse("/dashboard/product?added=1", status_code=303)



@router.get("/product", response_class=HTMLResponse)
async def products_list(request: Request, db: AsyncSession = Depends(get_db)):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    t = (request.query_params.get("type") or "").lower()
    if t not in ("", "all", "sell", "rent"):
        t = ""
    current_type = "all" if t in ("", "all") else t

    having = ""
    if current_type == "sell":
        having = "HAVING COALESCE(SUM(CASE WHEN pv.is_rental = FALSE THEN 1 END), 0) > 0"
    elif current_type == "rent":
        having = "HAVING COALESCE(SUM(CASE WHEN pv.is_rental = TRUE  THEN 1 END), 0) > 0"

    q = f"""
    SELECT 
        p.id,
        p.name,
        COUNT(pv.id) AS variant_count,
        COALESCE(SUM(CASE WHEN pv.is_rental THEN 1 ELSE 0 END), 0)      AS rental_count,
        COALESCE(SUM(CASE WHEN NOT pv.is_rental THEN 1 ELSE 0 END), 0)  AS sell_count,
        MAX(pv.image_url) FILTER (WHERE pv.image_url IS NOT NULL)        AS image_url,
        COALESCE(BOOL_OR(pv.is_active), FALSE)                           AS any_active
    FROM products p
    LEFT JOIN product_variants pv ON pv.product_id = p.id
    WHERE p.tenant_id = :tid
    GROUP BY p.id, p.name
    {having}
    ORDER BY p.name ASC
    LIMIT 300
"""
    res = await db.execute(text(q), {"tid": tenant_id})
    rows = res.fetchall() or []

    products = [
    {
        "product_id": r[0],
        "product": r[1],
        "variant_count": r[2] or 0,
        "rental_count": r[3] or 0,
        "sell_count": r[4] or 0,
        "image_url": r[5],
        "is_active": bool(r[6]),   # <— NEW
    } for r in rows
]


    tenant_name = request.session.get("tenant_name") or await get_tenant_name(tenant_id, db)
    return templates.TemplateResponse(
        "product_list.html",
        {
            "request": request,
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "products": products,
            "current_type": current_type,
        },
    )

@router.get("/product/{product_id}", response_class=HTMLResponse, name="product_detail")
async def product_detail(request: Request, product_id: int, db: AsyncSession = Depends(get_db)):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    res = await db.execute(text("""
        SELECT id, name
        FROM products
        WHERE id = :pid AND tenant_id = :tid
        LIMIT 1
    """), {"pid": product_id, "tid": tenant_id})
    prod = res.fetchone()
    if not prod:
        raise HTTPException(status_code=404, detail="Product not found")
    product = {"id": prod[0], "name": prod[1]}

    t = (request.query_params.get("type") or "").lower()
    if t not in ("", "all", "sell", "rent"):
        t = ""
    current_type = "all" if t in ("", "all") else t

    where = "WHERE pv.product_id = :pid"
    params = {"pid": product_id}
    if current_type == "rent":
        where += " AND pv.is_rental = TRUE"
    elif current_type == "sell":
        where += " AND pv.is_rental = FALSE"

    res = await db.execute(text(f"""
        SELECT pv.id, pv.color, pv.size, pv.fabric,
               pv.price, pv.rental_price, pv.available_stock, pv.is_rental
        FROM product_variants pv
        {where}
        ORDER BY pv.color, pv.size, pv.fabric
    """), params)
    rows = res.fetchall() or []

    variants = [
        {
            "variant_id": r[0],
            "color": r[1],
            "size": r[2],
            "fabric": r[3],
            "price": r[4],
            "rental_price": r[5],
            "available_stock": r[6],
            "is_rental": bool(r[7]),
        } for r in rows
    ]

    tenant_name = request.session.get("tenant_name") or await get_tenant_name(tenant_id, db)
    return templates.TemplateResponse(
        "product_detail.html",
        {
            "request": request,
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "product": product,
            "variants": variants,
            "current_type": current_type,
        },
    )




# --- PRODUCT: Modify (GET) ----------------------------------------------
@router.get("/product/{product_id}/modify")
async def product_modify_form(request: Request, product_id: int, db: AsyncSession = Depends(get_db)):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    res = await db.execute(text("""
        SELECT id, name, category, description, product_url
        FROM products
        WHERE id = :pid AND tenant_id = :tid
        LIMIT 1
    """), {"pid": product_id, "tid": tenant_id})
    row = res.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Product not found")

    p = {
        "id": row["id"],
        "product": row["name"],
        "category": row["category"],
        "description": row["description"],
        "product_url": row["product_url"]
    }
    return templates.TemplateResponse("product_modify.html", {"request": request, "p": p, "form_error": None})


# DELETE (AJAX-friendly)
@router.delete("/product/{product_id}/delete")
async def dashboard_delete_product_api(
    request: Request,
    product_id: int,
    db: AsyncSession = Depends(get_db),
):
    # tenant login check (same as your other dashboard actions)
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    result = await delete_product_everywhere(db=db, tenant_id=tenant_id, product_id=product_id)
    return JSONResponse(result)


# Optional: POST (for form submissions or <button formmethod="post">)
@router.post("/product/{product_id}/delete")
async def dashboard_delete_product_form(
    request: Request,
    product_id: int,
    db: AsyncSession = Depends(get_db),
):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    await delete_product_everywhere(db=db, tenant_id=tenant_id, product_id=product_id)
    # Redirect back to product list after deletion
    return RedirectResponse(url="/dashboard/products", status_code=303)



@router.post("/product/{product_id}/variants/{variant_id}/delete")
async def product_variant_delete(
    request: Request,
    product_id: int,
    variant_id: int,
    db: AsyncSession = Depends(get_db),
):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    try:
        await delete_variant_everywhere(db, tenant_id, product_id, variant_id)
        return RedirectResponse(
            f"/dashboard/product/{product_id}/variants?deleted=1",
            status_code=303
        )
    except HTTPException as e:
        # surface a concise reason in the UI
        detail = str(e.detail) if getattr(e, "detail", None) else "delete failed"
        return RedirectResponse(
            f"/dashboard/product/{product_id}/variants?err=delete&why={quote_plus(detail)}",
            status_code=303
        )

# --- PRODUCT: Modify (POST) ---------------------------------------------
@router.post("/product/{product_id}/modify")
async def product_modify_submit(
    request: Request,
    product_id: int,
    db: AsyncSession = Depends(get_db),
    product: str = Form(...),
    category: str = Form(""),
    description: str = Form(""),
    product_url: str = Form(""),
):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    # ensure ownership
    ok = await db.execute(text("SELECT 1 FROM products WHERE id=:pid AND tenant_id=:tid"), {"pid": product_id, "tid": tenant_id})
    if not ok.first():
        raise HTTPException(status_code=404, detail="Product not found")

    try:
        await db.execute(text("""
            UPDATE products
            SET name = :name,
                category = :category,
                description = :description,
                product_url = :product_url,
                updated_at = NOW()
            WHERE id = :pid AND tenant_id = :tid
        """), {
            "name": product.strip(),
            "category": category.strip() or None,
            "description": description.strip() or None,
            "product_url": product_url.strip() or None,
            "pid": product_id, "tid": tenant_id
        })
        await db.commit()

        # ✅ Pinecone: push product fields to ALL variants of this product
        try:
            await push_product_metadata_to_pinecone(
                db,
                product_id,
                product=product.strip(),
                category=(category or "").strip() or None,
                description=(description or "").strip() or None,
                product_url=(product_url or "").strip() or None,
            )
        except Exception:
            pass
    except Exception:
        await db.rollback()
        return templates.TemplateResponse(
            "product_modify.html",
            {"request": request, "p": {
                "id": product_id, "product": product, "category": category,
                "description": description, "product_url": product_url
            }, "form_error": "Failed to update product. Please try again."},
            status_code=400
        )

    return RedirectResponse("/dashboard/product?updated=1", status_code=303)


# ===================== VARIANTS =====================

# --- VARIANTS: List + optional edit panel (GET) -------------------------
@router.get("/product/{product_id}/variants")
async def product_variants_page(
    request: Request,
    product_id: int,
    variant_id: int | None = None,
    db: AsyncSession = Depends(get_db)
):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    # product header
    pres = await db.execute(text("""
        SELECT id, name FROM products WHERE id=:pid AND tenant_id=:tid
    """), {"pid": product_id, "tid": tenant_id})
    product_row = pres.mappings().first()
    if not product_row:
        raise HTTPException(status_code=404, detail="Product not found")

    # all variants
    vres = await db.execute(text("""
        SELECT id, color, size, fabric, price, rental_price, available_stock,
               is_rental, image_url, is_active, product_url, created_at, updated_at
        FROM product_variants
        WHERE product_id = :pid
        ORDER BY id DESC
    """), {"pid": product_id})
    variants = [dict(r) for r in vres.mappings().all()]

    # variant to edit (if any)
    edit_variant = None
    if variant_id:
        eres = await db.execute(text("""
            SELECT id, color, size, fabric, price, rental_price, available_stock,
                   is_rental, image_url, is_active, product_url
            FROM product_variants
            WHERE id = :vid AND product_id = :pid
            LIMIT 1
        """), {"vid": variant_id, "pid": product_id})
        edit_variant = eres.mappings().first()

    ctx = {
        "request": request,
        "product": {"id": product_row["id"], "name": product_row["name"]},
        "variants": variants,
        "edit": edit_variant,
        "form_error": None
    }
    return templates.TemplateResponse("product_variants.html", ctx)


# --- VARIANTS: Add (POST) -----------------------------------------------
@router.post("/product/{product_id}/variants/add")
async def product_variant_add(
    request: Request, product_id: int, db: AsyncSession = Depends(get_db),
    color: str = Form(""), size: str = Form(""), fabric: str = Form(""),
    price: float | None = Form(None), rental_price: float | None = Form(None),
    available_stock: int = Form(0), is_rental: bool = Form(False),
    image_url: str = Form(""), is_active: bool = Form(True),
    product_url: str = Form(""),
):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    # product belongs to tenant?
    ok = await db.execute(text("SELECT 1 FROM products WHERE id=:pid AND tenant_id=:tid"),
                          {"pid": product_id, "tid": tenant_id})
    if not ok.first():
        raise HTTPException(status_code=404, detail="Product not found")

    try:
        # INSERT + RETURNING id
        res = await db.execute(text("""
            INSERT INTO product_variants
              (product_id, color, size, fabric, price, rental_price, available_stock,
               is_rental, image_url, is_active, product_url)
            VALUES
              (:pid, :color, :size, :fabric, :price, :rental_price, :stock,
               :is_rental, :image_url, :is_active, :product_url)
            RETURNING id
        """), {
            "pid": product_id,
            "color": color or None,
            "size": size or None,
            "fabric": fabric or None,
            "price": price,
            "rental_price": rental_price,
            "stock": available_stock or 0,
            "is_rental": bool(is_rental),
            "image_url": image_url or None,
            "is_active": bool(is_active),
            "product_url": product_url or None
        })
        new_variant_id = res.scalar_one()
        await db.commit()

        # ✅ Pinecone: push this variant’s metadata
        try:
            await upsert_variant_image_from_db(db, new_variant_id)
            push_variant_metadata_to_pinecone(
                new_variant_id,
                color=color or None,
                size=size or None,
                fabric=fabric or None,
                price=price,
                rental_price=rental_price,
                available_stock=available_stock or 0,
                is_rental=bool(is_rental),
                image_url=image_url or None,
                is_active=bool(is_active),
                product_url=product_url or None,
            )
        except Exception:
            pass

    except Exception:
        await db.rollback()
        return RedirectResponse(f"/dashboard/product/{product_id}/variants?err=add", status_code=303)

    return RedirectResponse(f"/dashboard/product/{product_id}/variants?added=1", status_code=303)

# --- VARIANTS: Update (POST) --------------------------------------------
@router.post("/product/{product_id}/variants/{variant_id}/edit")
async def product_variant_edit(
    request: Request, product_id: int, variant_id: int,
    db: AsyncSession = Depends(get_db),
    color: str = Form(""), size: str = Form(""), fabric: str = Form(""),
    price: float | None = Form(None), rental_price: float | None = Form(None),
    available_stock: int = Form(0), is_rental: bool = Form(False),
    image_url: str = Form(""), is_active: bool = Form(True),
    product_url: str = Form(""),
):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    # 1) OWNERSHIP CHECKS (don’t skip these)
    ok = await db.execute(
        text("SELECT 1 FROM products WHERE id=:pid AND tenant_id=:tid"),
        {"pid": product_id, "tid": tenant_id},
    )
    if not ok.first():
        raise HTTPException(status_code=404, detail="Product not found")

    prev = await db.execute(
        text("SELECT image_url FROM product_variants WHERE id=:vid AND product_id=:pid"),
        {"vid": variant_id, "pid": product_id},
    )
    row = prev.first()
    if not row:
        raise HTTPException(status_code=404, detail="Variant not found")
    prev_image_url = row[0] or ""

    # 2) UPDATE DB
    try:
        await db.execute(text("""
            UPDATE product_variants
               SET color=:color, size=:size, fabric=:fabric,
                   price=:price, rental_price=:rental_price,
                   available_stock=:stock, is_rental=:is_rental,
                   image_url=:image_url, is_active=:is_active,
                   product_url=:product_url, updated_at = NOW()
             WHERE id=:vid AND product_id=:pid
        """), {
            "color": color or None, "size": size or None, "fabric": fabric or None,
            "price": price, "rental_price": rental_price, "stock": available_stock or 0,
            "is_rental": bool(is_rental), "image_url": (image_url or None),
            "is_active": bool(is_active), "product_url": product_url or None,
            "vid": variant_id, "pid": product_id
        })
        await db.commit()

        # 3) SYNC PINECONE
        try:
            # Only re-embed if the image_url actually changed (saves time)
            if (prev_image_url or "") != (image_url or ""):
                await upsert_variant_image_from_db(db, variant_id)
            else:
                # keep metadata fresh even if embedding unchanged
                push_variant_metadata_to_pinecone(
                    variant_id,
                    color=color or None,
                    size=size or None,
                    fabric=fabric or None,
                    price=price,
                    rental_price=rental_price,
                    available_stock=available_stock or 0,
                    is_rental=bool(is_rental),
                    image_url=image_url or None,
                    is_active=bool(is_active),
                    product_url=product_url or None,
                )
        except Exception:
            # don’t block the UI if Pinecone hiccups
            pass

    except Exception:
        await db.rollback()
        return RedirectResponse(
            f"/dashboard/product/{product_id}/variants?err=edit&variant_id={variant_id}",
            status_code=303
        )

    return RedirectResponse(f"/dashboard/product/{product_id}/variants?updated=1", status_code=303)


# --- VARIANTS: Delete (POST) --------------------------------------------
@router.post("/product/{product_id}/variants/{variant_id}/delete")
async def product_variant_delete(request: Request, product_id: int, variant_id: int, db: AsyncSession = Depends(get_db)):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    # ownership + existence
    ok = await db.execute(text("""
        SELECT 1 FROM products p
        WHERE p.id=:pid AND p.tenant_id=:tid
    """), {"pid": product_id, "tid": tenant_id})
    if not ok.first():
        raise HTTPException(status_code=404, detail="Product not found")

    vok = await db.execute(text("""
        SELECT 1 FROM product_variants WHERE id=:vid AND product_id=:pid
    """), {"vid": variant_id, "pid": product_id})
    if not vok.first():
        raise HTTPException(status_code=404, detail="Variant not found")

    await db.execute(text("DELETE FROM product_variants WHERE id=:vid AND product_id=:pid"),
                     {"vid": variant_id, "pid": product_id})
    await db.commit()
    return RedirectResponse(f"/dashboard/product/{product_id}/variants?deleted=1", status_code=303)



# ---------------------------------------------------------------------
# Orders & Rentals
# ---------------------------------------------------------------------
@router.get("/orders", response_class=HTMLResponse)
async def orders_list(request: Request, db: AsyncSession = Depends(get_db)):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    res = await db.execute(text("""
        SELECT 
            o.id,
            o.order_number,
            o.order_type,           -- 'purchase' | 'rental'
            o.status,
            o.payment_status,
            o.price        AS amount,
            o.discount,
            o.start_date,
            o.end_date,
            o.created_at,

            c.id           AS customer_id,
            c.name         AS customer_name,
            c.phone        AS customer_phone,
            c.email        AS customer_email,

            pv.id          AS variant_id,
            pv.id::text    AS variant_label,

            p.id           AS product_id,
            p.name         AS product_name,
            p.category     AS product_category
        FROM orders o
        JOIN customers        c  ON c.id  = o.customer_id
        JOIN product_variants pv ON pv.id = o.product_variant_id
        JOIN products         p  ON p.id  = pv.product_id
        WHERE o.tenant_id = :tid
        ORDER BY o.created_at DESC
        LIMIT 2000
    """), {"tid": tenant_id})
    rows = res.fetchall() or []

    orders = [{
        "id": r[0],
        "order_number": r[1],
        "order_type": r[2],
        "status": r[3],
        "payment_status": r[4],
        "amount": r[5],
        "discount": r[6],
        "start_date": r[7],
        "end_date": r[8],
        "created_at": r[9],
        "customer_id": r[10],
        "customer_name": r[11],
        "customer_phone": r[12],
        "customer_email": r[13],
        "variant_id": r[14],
        "variant_label": r[15],
        "product_id": r[16],
        "product_name": r[17],
        "product_category": r[18],
        "is_rental": (r[2] == "rental"),
    } for r in rows]

    tenant_name = request.session.get("tenant_name") or await get_tenant_name(tenant_id, db)
    return templates.TemplateResponse(
        "orders_list.html",
        {
            "request": request,
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "orders": orders,
        },
    )

@router.get("/rentals", response_class=HTMLResponse)
async def rentals_list(request: Request, db: AsyncSession = Depends(get_db)):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    s = (request.query_params.get("status") or "").lower()
    allowed = {"", "all", "active", "returned", "cancelled"}
    if s not in allowed:
        s = ""
    current_status = "all" if s in ("", "all") else s

    where_status = ""
    params = {"tid": tenant_id}
    if current_status != "all":
        where_status = "AND r.status::text = :status"
        params["status"] = current_status

    res = await db.execute(text(f"""
        SELECT
            r.id,
            r.customer_id,
            c.name       AS customer_name,
            c.phone      AS customer_phone,
            c.email      AS customer_email,

            pv.id        AS variant_id,
            pv.id::text  AS variant_label,

            p.id         AS product_id,
            p.name       AS product_name,
            p.category   AS product_category,

            r.rental_start_date,
            r.rental_end_date,
            r.rental_price,
            r.status,
            r.created_at
        FROM rentals r
        JOIN customers        c  ON c.id  = r.customer_id
        JOIN product_variants pv ON pv.id = r.product_variant_id
        JOIN products         p  ON p.id  = pv.product_id
        WHERE r.tenant_id = :tid
          {where_status}
        ORDER BY r.created_at DESC
        LIMIT 2000
    """), params)
    rows = res.fetchall() or []

    rentals = [{
        "id": r[0],
        "customer_id": r[1],
        "customer_name": r[2],
        "customer_phone": r[3],
        "customer_email": r[4],
        "variant_id": r[5],
        "variant_label": r[6],
        "product_id": r[7],
        "product_name": r[8],
        "product_category": r[9],
        "start_date": r[10],
        "end_date": r[11],
        "rental_price": r[12],
        "status": r[13],
        "created_at": r[14],
    } for r in rows]

    tenant_name = request.session.get("tenant_name") or await get_tenant_name(tenant_id, db)
    return templates.TemplateResponse(
        "rentals_list.html",
        {
            "request": request,
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "rentals": rentals,
            "current_status": current_status,
        },
    )

# ---------------------------------------------------------------------
# Chat history — ONLY user images fetch via Meta; assistant uses DB
# ---------------------------------------------------------------------
def _normalize_user_media(m: Dict[str, Any]) -> Dict[str, Any]:
    """
    For incoming ('in') messages only:
    - Resolve a working image URL through our proxy (preferred),
      or from meta.image_url / meta.image.url / meta.image.
    - If text is just a lookaside/graph URL and we show an image,
      hide that text to avoid duplicate long links.
    """
    if not isinstance(m, dict) or m.get("direction") != "in":
        return m

    meta = (m.get("meta") or {})
    txt  = (m.get("text") or "").strip()

    # 1) already resolved?
    img = meta.get("proxy_url")

    # 2) DB-style fields
    if not img:
        if isinstance(meta.get("image_url"), str):
            img = meta["image_url"]
        elif isinstance(meta.get("image"), dict) and isinstance(meta["image"].get("url"), str):
            img = meta["image"]["url"]
        elif isinstance(meta.get("image"), str):
            img = meta["image"]

    # 3) Extract media_id from pasted lookaside link if needed
    if not img and txt:
        mm = MID_RE.search(txt)
        if mm:
            meta["proxy_url"] = f"/dashboard/media/wa/{mm.group(1)}"
            img = meta["proxy_url"]

    # 4) If it's a WA URL and no explicit proxy yet, use proxy-by-url
    if img and not meta.get("proxy_url"):
        low = img.lower()
        if "lookaside.fbsbx.com" in low or "graph.facebook.com" in low:
            meta["proxy_url"] = f"/dashboard/media/wa?url={quote_plus(img)}"
            img = meta["proxy_url"]

    meta["resolved_image"] = img
    m["meta"] = meta

    # Hide the long WA link if we rendered its image
    if img and txt:
        low = txt.lower()
        if "lookaside.fbsbx.com" in low or "graph.facebook.com" in low:
            m["text"] = ""

    return m

@router.get("/chat/{customer_id}", response_class=HTMLResponse)
async def chat_history(request: Request, customer_id: int, db: AsyncSession = Depends(get_db)):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    res = await db.execute(text("""
        SELECT id, started_at, ended_at, transcript
        FROM chat_sessions
        WHERE customer_id = :cid
        ORDER BY started_at DESC
        LIMIT 1
    """), {"cid": customer_id})
    row = res.fetchone()

    if not row:
        session = None
        messages: List[Dict[str, Any]] = []
    else:
        session = {"id": row[0], "started_at": row[1], "ended_at": row[2]}
        transcript = row[3]
        # transcript may be JSONB (list) or a JSON string
        if isinstance(transcript, str):
            try:
                transcript = json.loads(transcript)
            except Exception:
                transcript = []
        messages = [_normalize_user_media(m) if isinstance(m, dict) else m for m in (transcript or [])]

    tenant_name = request.session.get("tenant_name") or await get_tenant_name(tenant_id, db)

    return templates.TemplateResponse(
        "chat_history.html",
        {
            "request": request,
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "customer_id": customer_id,
            "session": session,
            "messages": messages,
        },
    )
