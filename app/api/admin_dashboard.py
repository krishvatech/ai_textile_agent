# app/api/admin_dashboard.py
from __future__ import annotations
from fastapi import Form, status

from typing import Any, Dict, List, Optional, OrderedDict as _OrderedDict
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
    tenants_count = await _scalar_int(
    db,
    "SELECT COUNT(DISTINCT tenant_id) FROM public.users "
    "WHERE role = 'tenant_admin' AND tenant_id IS NOT NULL AND COALESCE(is_active, true) = true"
    )
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

@router.get("/tenants", response_class=HTMLResponse)
async def admin_tenants(
    request: Request,
    show_inactive: int = 0,                    # 0 = only active (default), 1 = include inactive
    db: AsyncSession = Depends(get_db),
):
    guard = require_superadmin(request)
    if isinstance(guard, RedirectResponse):
        return guard

    res = await db.execute(text("""
        SELECT
            t.id,
            t.name,
            t.whatsapp_number,
            t.address,
            t.is_active,
            COUNT(DISTINCT c.id) AS customers_count,
            (
              SELECT MIN(u.email)  -- first/primary admin email (deterministic)
              FROM public.users u
              WHERE u.tenant_id = t.id
                AND u.role = 'tenant_admin'
                AND COALESCE(u.is_active, true) = true
            ) AS admin_email
        FROM tenants t
        LEFT JOIN customers c ON c.tenant_id = t.id
        WHERE EXISTS (
            SELECT 1
            FROM public.users u2
            WHERE u2.tenant_id = t.id
              AND u2.role = 'tenant_admin'
              AND COALESCE(u2.is_active, true) = true
        )
          AND (:include_inactive = 1 OR COALESCE(t.is_active, true) = true)
        GROUP BY t.id, t.name, t.whatsapp_number, t.address, t.is_active
        ORDER BY t.name
    """), {"include_inactive": int(show_inactive)})

    rows = res.fetchall() or []

    tenants = [{
        "id": int(r[0]),
        "name": r[1] or f"Tenant {r[0]}",
        "whatsapp_number": (r[2] or "").strip(),
        "address": (r[3] or "").strip(),
        "is_active": bool(r[4]) if r[4] is not None else False,
        "customers_count": int(r[5] or 0),
        "admin_email": (r[6] or "").strip(),
    } for r in rows]

    return templates.TemplateResponse("tenant_list.html", {
        "request": request,
        "tenants": tenants,
        "include_inactive": bool(show_inactive),
    })


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
    
    cust_by_tenant = await _series_customers_by_tenant_monthly(db, top_n=5, months=None)
    tenants_created = await _series_tenants_created_monthly(db, months=None)

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

        # NEW: charts
        "cust_by_tenant": cust_by_tenant,   # {labels:[], datasets:[{label, data}]}
        "tenants_created": tenants_created, # {labels:[], data:[]}
    }
    return templates.TemplateResponse("dashboard.html", ctx)


@router.get("/customers", response_class=HTMLResponse)
async def admin_customers(request: Request, tenant_id: Optional[int] = None,
                          db: AsyncSession = Depends(get_db)):
    guard = require_superadmin(request)
    if isinstance(guard, RedirectResponse):
        return guard

    # If a tenant_id is provided, fetch its name for the page heading
    selected_tenant_name = None
    if tenant_id:
        r = await db.execute(text("SELECT name FROM tenants WHERE id = :tid LIMIT 1"), {"tid": tenant_id})
        selected_tenant_name = r.scalar()

    where = "WHERE c.tenant_id = :tid" if tenant_id else ""
    params: Dict[str, Any] = {"tid": tenant_id} if tenant_id else {}

    res = await db.execute(text(f"""
        SELECT c.id, c.whatsapp_id, c.name, c.preferred_language, c.phone, c.email,
               c.created_at, c.is_active, c.loyalty_points, c.tenant_id,
               t.name AS tenant_name
        FROM customers c
        LEFT JOIN tenants t ON t.id = c.tenant_id
        {where}
        ORDER BY c.created_at DESC
        LIMIT 500
    """), params)
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

    # Fallback: if tenant has 0 customers, still show its name if we looked it up
    if not selected_tenant_name and customers and tenant_id:
        selected_tenant_name = customers[0]["tenant_name"]

    return templates.TemplateResponse(
        "customer_list.html",
        {
            "request": request,
            "customers": customers,
            "tenant_scope": "single" if tenant_id else "all",
            "selected_tenant_id": tenant_id,
            "selected_tenant_name": selected_tenant_name,
            "tenant_name": "Superadmin",  # shows in the topbar subtitle
        },
    )


# ----------------------------------------------------------------------------

# ── Add Tenant: form ──────────────────────────────────────────────────────────
@router.get("/tenants/add", response_class=HTMLResponse)
async def tenant_add_form(request: Request):
    guard = require_superadmin(request)
    if isinstance(guard, RedirectResponse):
        return guard
    return templates.TemplateResponse("tenant_add.html", {
        "request": request,
        "tenant_name": "Superadmin",
        "form_error": None,
    })


# ── Add Tenant: submit ────────────────────────────────────────────────────────
@router.post("/tenants/add")
async def tenant_add_submit(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    whatsapp_number: str = Form(""),
    phone_number: str = Form(""),
    address: str = Form(""),
    language: str = Form("en-IN"),
    is_active: bool = Form(True),
    db: AsyncSession = Depends(get_db),
):
    guard = require_superadmin(request)
    if isinstance(guard, RedirectResponse):
        return guard

    # Basic sanity (very light; keep your existing style)
    if not name.strip():
        return templates.TemplateResponse("tenant_add.html", {
            "request": request,
            "tenant_name": "Superadmin",
            "form_error": "Name is required.",
        })
    if not email.strip():
        return templates.TemplateResponse("tenant_add.html", {
            "request": request,
            "tenant_name": "Superadmin",
            "form_error": "Email is required.",
        })

    try:
        # 1) create tenant and get id
        insert_tenant = text("""
            INSERT INTO tenants
            (name, whatsapp_number, phone_number, address, language, is_active, email, password, created_at, updated_at)
            VALUES
            (:name, :whatsapp_number, :phone_number, :address, :language, :is_active, :email, :password, NOW(), NOW())
            RETURNING id
        """)
        res = await db.execute(insert_tenant, {
            "name": name.strip(),
            "whatsapp_number": whatsapp_number.strip() or None,
            "phone_number": phone_number.strip() or None,
            "address": address.strip() or None,
            "language": language.strip() or None,
            "is_active": bool(is_active),
            "email": email.strip(),
            "password": password,   # your schema uses plain 'password' on tenants
        })
        tenant_id = int(res.scalar())

        # 2) create tenant_admin user (mirrors your current data style)
        insert_user = text("""
            INSERT INTO public.users
            (email, hashed_password, role, tenant_id, is_active, created_at, updated_at)
            VALUES
            (:email, :hashed_password, 'tenant_admin', :tenant_id, TRUE, NOW(), NOW())
        """)
        await db.execute(insert_user, {
            "email": email.strip(),
            "hashed_password": password,   # your table stores plain values currently
            "tenant_id": tenant_id,
        })

        await db.commit()
        return RedirectResponse("/admin/tenants?created=1", status_code=status.HTTP_303_SEE_OTHER)

    except Exception as e:
        await db.rollback()
        return templates.TemplateResponse("tenant_add.html", {
            "request": request,
            "tenant_name": "Superadmin",
            "form_error": f"Could not save tenant: {e}",
        })


# ----------------------------------------------------------------------------

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



# ── Customers over time by tenant (top N) ─────────────────────────────────────
async def _series_customers_by_tenant_monthly(db: AsyncSession, top_n: int = 5, months: int | None = None):
    # Pick top N tenants by total customers (all time)
    res = await db.execute(text("""
        SELECT t.id, COALESCE(t.name, 'Tenant ' || t.id::text) AS name, COUNT(c.id) AS cnt
        FROM tenants t
        JOIN customers c ON c.tenant_id = t.id
        GROUP BY t.id, t.name
        ORDER BY cnt DESC, name ASC
        LIMIT :top_n
    """), {"top_n": int(top_n)})
    top = res.fetchall() or []
    if not top:
        return {"labels": [], "datasets": []}

    top_ids  = [int(r[0]) for r in top]
    top_names = {int(r[0]): (r[1] or f"Tenant {r[0]}") for r in top}

    # Build a safe IN clause like (:id0, :id1, ...)
    id_params = {f"id{i}": tid for i, tid in enumerate(top_ids)}
    in_clause = ", ".join([f":id{i}" for i in range(len(top_ids))])

    # Optional month filter
    month_filter = ""
    if months:
        month_filter = f"AND c.created_at >= NOW() - INTERVAL '{int(months)} months'"

    # Pull monthly counts for those tenants
    res = await db.execute(text(f"""
        SELECT DATE_TRUNC('month', c.created_at) AS mth, c.tenant_id, COUNT(*) AS cnt
        FROM customers c
        WHERE c.tenant_id IN ({in_clause})
          {month_filter}
        GROUP BY mth, c.tenant_id
        ORDER BY mth
    """), id_params)
    rows = res.fetchall() or []

    # Collect all months in order
    from collections import OrderedDict, defaultdict
    months_map = OrderedDict()
    per_tenant = defaultdict(dict)  # {tenant_id: {mth: cnt}}

    for mth, tid, cnt in rows:
        months_map[mth] = True
        per_tenant[int(tid)][mth] = int(cnt)

    labels = [m.strftime("%b %Y") for m in months_map.keys()]
    datasets = []
    for tid in top_ids:
        data = [per_tenant.get(tid, {}).get(m, 0) for m in months_map.keys()]
        datasets.append({"label": top_names[tid], "data": data})

    return {"labels": labels, "datasets": datasets}


# ── Tenants created per month ─────────────────────────────────────────────────
async def _series_tenants_created_monthly(db: AsyncSession, months: int | None = None):
    month_filter = ""
    if months:
        month_filter = f"WHERE t.created_at >= NOW() - INTERVAL '{int(months)} months'"

    res = await db.execute(text(f"""
        SELECT DATE_TRUNC('month', t.created_at) AS mth, COUNT(*) AS cnt
        FROM tenants t
        {month_filter}
        GROUP BY mth
        ORDER BY mth
    """))
    rows = res.fetchall() or []

    labels = [r[0].strftime("%b %Y") for r in rows]
    data   = [int(r[1] or 0) for r in rows]
    return {"labels": labels, "data": data}



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