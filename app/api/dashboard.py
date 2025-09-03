# app/api/dashboard.py
from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

# Use the same get_db that admin.py relies on
from app.db.session import get_db  # <-- same pattern as admin.py

router = APIRouter()
templates = Jinja2Templates(directory="app/templates/tenants")

ALLOWED_DASHBOARD_ROLES = {"tenant_admin", "superadmin"}

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

# ---------- Customers ----------
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

# ---------- Products ----------
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
            MAX(pv.image_url) FILTER (WHERE pv.image_url IS NOT NULL)        AS image_url
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

# --- Chat history page ---
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
        messages = transcript or []

    tenant_name = request.session.get("tenant_name") or await get_tenant_name(tenant_id, db)

    return templates.TemplateResponse(
        "chat_history.html",
        {"request": request, "tenant_id": tenant_id, "tenant_name": tenant_name, "customer_id": customer_id, "session": session, "messages": messages},
    )

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
