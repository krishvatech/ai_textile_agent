# app/api/admin_dashboard.py
from __future__ import annotations

from typing import Any, Dict, List, OrderedDict as _OrderedDict
from collections import OrderedDict

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

# keep the same import convention as your other routers
from app.db.session import get_db

router = APIRouter()
templates = Jinja2Templates(directory="app/templates/admin")


# ──────────────────────────────────────────────────────────────────────────────
# Auth guard: superadmin only
# ──────────────────────────────────────────────────────────────────────────────
def require_superadmin(request: Request):
    """
    Only allow role='superadmin'. If not authenticated, redirect to /login
    with a 'next' param using the current path (same pattern as login/dashboard).
    """
    role = request.session.get("role")
    if role != "superadmin":
        nxt = request.url.path or "/admin"
        return RedirectResponse(url=f"/login?next={nxt}", status_code=303)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (SQL snippets are intentionally simple and portable to your schema)
# ──────────────────────────────────────────────────────────────────────────────
async def _scalar_int(db: AsyncSession, sql: str, params: Dict[str, Any] | None = None) -> int:
    res = await db.execute(text(sql), params or {})
    return int(res.scalar() or 0)


async def _kpis_all_tenants(db: AsyncSession) -> Dict[str, Any]:
    """
    Collect global KPIs across all tenants. Matches your admin dashboard tiles.
    """
    tenants_count   = await _scalar_int(db, "SELECT COUNT(*) FROM tenants")
    users_count     = await _scalar_int(db, "SELECT COUNT(*) FROM public.users")
    customers_count = await _scalar_int(db, "SELECT COUNT(*) FROM customers")
    products_count  = await _scalar_int(db, "SELECT COUNT(*) FROM products")
    orders_count    = await _scalar_int(db, "SELECT COUNT(*) FROM orders")
    rentals_count   = await _scalar_int(db, "SELECT COUNT(*) FROM rentals")

    # Selling vs Rental product counts via product_variants.is_rental
    selling_products_count = await _scalar_int(
        db, "SELECT COUNT(*) FROM product_variants WHERE COALESCE(is_rental, false) = false"
    )
    rental_products_count = await _scalar_int(
        db, "SELECT COUNT(*) FROM product_variants WHERE COALESCE(is_rental, false) = true"
    )

    # Optional revenue (safe even if columns are absent → 0)
    res = await db.execute(text("SELECT COALESCE(SUM(price), 0) FROM orders"))
    revenue_orders = float(res.scalar() or 0.0)
    res = await db.execute(text("SELECT COALESCE(SUM(rental_price), 0) FROM rentals"))
    revenue_rentals = float(res.scalar() or 0.0)

    return {
        "tenants_count": tenants_count,
        "users_count": users_count,
        "customers_count": customers_count,
        "products_count": products_count,
        "orders_count": orders_count,
        "rentals_count": rentals_count,
        "selling_products_count": selling_products_count,
        "rental_products_count": rental_products_count,
        "revenue_orders": revenue_orders,
        "revenue_rentals": revenue_rentals,
        "revenue_total": revenue_orders + revenue_rentals,
    }


async def _series_monthly_all(db: AsyncSession) -> Dict[str, List[Any]]:
    """
    Build monthly series for Orders (sell) and Rentals (rent) for the chart.
    Uses created_at for orders, and rental_start_date (fallback to created_at) for rentals.
    """
    # Orders per month
    res = await db.execute(text("""
        SELECT DATE_TRUNC('month', o.created_at) AS mth, COUNT(*) AS c
        FROM orders o
        GROUP BY mth
        ORDER BY mth
    """))
    sell_rows = res.fetchall() or []

    # Rentals per month
    res = await db.execute(text("""
        SELECT DATE_TRUNC('month', COALESCE(r.rental_start_date, r.created_at)) AS mth, COUNT(*) AS c
        FROM rentals r
        GROUP BY mth
        ORDER BY mth
    """))
    rent_rows = res.fetchall() or []

    merged: _OrderedDict[Any, Dict[str, int]] = OrderedDict()
    for m, c in sell_rows:
        merged[m] = {"sell": int(c or 0), "rent": 0}
    for m, c in rent_rows:
        merged.setdefault(m, {"sell": 0, "rent": 0})
        merged[m]["rent"] += int(c or 0)

    labels = [m.strftime("%b %Y") for m in merged.keys()]
    sell_series = [v["sell"] for v in merged.values()]
    rent_series = [v["rent"] for v in merged.values()]
    return {"labels": labels, "sell": sell_series, "rent": rent_series}


# ──────────────────────────────────────────────────────────────────────────────
# Views
# ──────────────────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse, name="admin_home")
async def admin_home(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Superadmin landing page → /admin
    Renders KPIs + monthly sell vs rent chart using admin/dashboard.html
    """
    guard = require_superadmin(request)
    if isinstance(guard, RedirectResponse):
        return guard

    kpis = await _kpis_all_tenants(db)
    series = await _series_monthly_all(db)

    # (Optional) Top 10 tenants by orders in last 30 days — template can show this if desired
    res = await db.execute(text("""
        SELECT t.id, t.name, COUNT(o.id) AS orders_30d
        FROM tenants t
        LEFT JOIN orders o
               ON o.tenant_id = t.id
              AND o.created_at >= NOW() - INTERVAL '30 days'
        GROUP BY t.id, t.name
        ORDER BY orders_30d DESC, t.name ASC
        LIMIT 10
    """))
    top_tenants = [{"tenant_id": r[0], "tenant_name": r[1], "orders_30d": int(r[2] or 0)}
                   for r in (res.fetchall() or [])]

    # Map to the exact variable names your templates expect
    ctx = {
        "request": request,
        "tenant_name": "Superadmin",
        "tenants_count":   kpis["tenants_count"],
        "customers_count": kpis["customers_count"],
        "products_count":  kpis["products_count"],
        "selling_products_count": kpis["selling_products_count"],
        "rental_products_count":  kpis["rental_products_count"],
        "orders_count":    kpis["orders_count"],
        "rentals_count":   kpis["rentals_count"],
        "chart_labels": series["labels"],
        "chart_sell":   series["sell"],
        "chart_rent":   series["rent"],
        "top_tenants":  top_tenants,
    }
    return templates.TemplateResponse("dashboard.html", ctx)


@router.get("/customers", response_class=HTMLResponse)
async def admin_customers(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Global customer list (latest 500 across all tenants) — uses admin/customer_list.html
    """
    guard = require_superadmin(request)
    if isinstance(guard, RedirectResponse):
        return guard

    res = await db.execute(text("""
        SELECT c.id, c.whatsapp_id, c.name, c.preferred_language, c.phone, c.email,
               c.created_at, c.is_active, c.loyalty_points, c.tenant_id,
               t.name AS tenant_name
        FROM customers c
        LEFT JOIN tenants t ON t.id = c.tenant_id
        ORDER BY c.created_at DESC
        LIMIT 500
    """))
    rows = res.fetchall() or []

    customers = [{
        "id": r[0],
        "whatsapp_id": r[1] or "-",
        "name": (r[2] or "(no name)"),
        "preferred_language": r[3] or "—",
        "phone": r[4] or "—",
        "email": r[5] or "—",
        "created_at": r[6],
        "is_active": bool(r[7]) if r[7] is not None else False,
        "loyalty_points": r[8] or 0,
        "tenant_id": r[9],
        "tenant_name": r[10] or "—",
    } for r in rows]

    return templates.TemplateResponse(
        "customer_list.html",
        {"request": request, "customers": customers, "tenant_scope": "all"},
    )


@router.get("/customer/{customer_id}", response_class=HTMLResponse)
async def admin_customer_detail(request: Request, customer_id: int, db: AsyncSession = Depends(get_db)):
    """
    Global customer detail — uses admin/customer_detail.html
    """
    guard = require_superadmin(request)
    if isinstance(guard, RedirectResponse):
        return guard

    # Basic profile
    res = await db.execute(text("""
        SELECT c.id, c.name, c.phone, c.email, c.preferred_language,
               c.loyalty_points, c.created_at, c.tenant_id, t.name AS tenant_name
        FROM customers c
        LEFT JOIN tenants t ON t.id = c.tenant_id
        WHERE c.id = :cid
        LIMIT 1
    """), {"cid": customer_id})
    row = res.fetchone()
    if not row:
        return HTMLResponse("<h3>Customer not found</h3>", status_code=404)

    customer = {
        "id": row[0],
        "name": row[1] or "(no name)",
        "phone": row[2] or "-",
        "email": row[3] or "-",
        "preferred_language": row[4] or "-",
        "loyalty_points": row[5] or 0,
        "created_at": row[6],
        "tenant_id": row[7],
        "tenant_name": row[8] or "—",
    }

    # Recent Orders (10)
    res = await db.execute(text("""
        SELECT o.id, o.order_type, o.status, o.price, o.created_at,
               pv.color, pv.size, pv.fabric, p.name
        FROM orders o
        JOIN product_variants pv ON pv.id = o.product_variant_id
        JOIN products p         ON p.id = pv.product_id
        WHERE o.customer_id = :cid
        ORDER BY o.created_at DESC
        LIMIT 10
    """), {"cid": customer_id})
    order_rows = res.fetchall() or []
    orders = [{
        "id": r[0],
        "order_type": r[1],
        "status": r[2],
        "price": r[3],
        "created_at": r[4],
        "variant": f"{r[8]} — {r[5]}/{r[6]}/{r[7]}",
    } for r in order_rows]

    # Recent Rentals (10)
    res = await db.execute(text("""
        SELECT r.id, r.status, r.rental_price, r.rental_start_date, r.rental_end_date, r.created_at,
               pv.color, pv.size, pv.fabric, p.name
        FROM rentals r
        JOIN product_variants pv ON pv.id = r.product_variant_id
        JOIN products p         ON p.id = pv.product_id
        WHERE r.customer_id = :cid
        ORDER BY r.created_at DESC
        LIMIT 10
    """), {"cid": customer_id})
    rent_rows = res.fetchall() or []
    rentals = [{
        "id": r[0],
        "status": r[1],
        "rental_price": r[2],
        "start_date": r[3],
        "end_date": r[4],
        "created_at": r[5],
        "variant": f"{r[9]} — {r[6]}/{r[7]}/{r[8]}",
    } for r in rent_rows]

    return templates.TemplateResponse(
        "customer_detail.html",
        {"request": request, "customer": customer, "orders": orders, "rentals": rentals},
    )


@router.get("/product/{product_id}", response_class=HTMLResponse)
async def admin_product_detail(request: Request, product_id: int, db: AsyncSession = Depends(get_db)):
    """
    Global product detail — uses admin/product_detail.html
    """
    guard = require_superadmin(request)
    if isinstance(guard, RedirectResponse):
        return guard

    res = await db.execute(text("""
        SELECT p.id, p.name, p.category, p.tenant_id, t.name as tenant_name
        FROM products p
        LEFT JOIN tenants t ON t.id = p.tenant_id
        WHERE p.id = :pid
        LIMIT 1
    """), {"pid": product_id})
    row = res.fetchone()
    if not row:
        return HTMLResponse("<h3>Product not found</h3>", status_code=404)

    product = {
        "id": row[0],
        "name": row[1],
        "category": row[2],
        "tenant_id": row[3],
        "tenant_name": row[4] or "—",
    }

    res = await db.execute(text("""
        SELECT pv.id, pv.color, pv.size, pv.fabric,
               pv.price, pv.rental_price, pv.available_stock, pv.is_rental
        FROM product_variants pv
        WHERE pv.product_id = :pid
        ORDER BY pv.color, pv.size, pv.fabric
    """), {"pid": product_id})
    rows = res.fetchall() or []

    variants = [{
        "variant_id": r[0],
        "color": r[1],
        "size": r[2],
        "fabric": r[3],
        "price": r[4],
        "rental_price": r[5],
        "available_stock": r[6],
        "is_rental": bool(r[7]),
    } for r in rows]

    return templates.TemplateResponse(
        "product_detail.html",
        {"request": request, "product": product, "variants": variants},
    )


# ──────────────────────────────────────────────────────────────────────────────
# JSON analytics endpoints used by charts/widgets
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/analytics/orders-vs-rentals-monthly", response_class=JSONResponse)
async def admin_analytics_monthly(db: AsyncSession = Depends(get_db)):
    """
    Returns { labels: [...], sell: [...], rent: [...] }
    """
    series = await _series_monthly_all(db)
    return JSONResponse(series)


@router.get("/analytics/top-tenants", response_class=JSONResponse)
async def admin_analytics_top_tenants(db: AsyncSession = Depends(get_db)):
    res = await db.execute(text("""
        SELECT t.id, t.name, COUNT(o.id) AS orders_30d
        FROM tenants t
        LEFT JOIN orders o
               ON o.tenant_id = t.id
              AND o.created_at >= NOW() - INTERVAL '30 days'
        GROUP BY t.id, t.name
        ORDER BY orders_30d DESC, t.name ASC
        LIMIT 10
    """))
    items = [{"tenant_id": r[0], "tenant_name": r[1], "orders_30d": int(r[2] or 0)} for r in (res.fetchall() or [])]
    return JSONResponse({"items": items})


# ──────────────────────────────────────────────────────────────────────────────
# Impersonate tenant → jump to tenant dashboard
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/impersonate/{tenant_id}")
async def admin_impersonate(request: Request, tenant_id: int, db: AsyncSession = Depends(get_db)):
    """
    Set session tenant and hop to /dashboard (tenant view).
    Keeps role='superadmin' (same session scheme used by /login).
    """
    guard = require_superadmin(request)
    if isinstance(guard, RedirectResponse):
        return guard

    # Validate tenant exists
    res = await db.execute(text("SELECT name FROM tenants WHERE id = :tid LIMIT 1"), {"tid": tenant_id})
    row = res.fetchone()
    if not row:
        return RedirectResponse("/admin", status_code=303)

    request.session["tenant_id"] = int(tenant_id)
    request.session["tenant_name"] = row[0] or f"Tenant {tenant_id}"
    return RedirectResponse("/dashboard", status_code=303)
