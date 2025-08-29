# app/api/dashboard.py
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any

from app.db.db_connection import get_db_connection, close_db_connection  # <-- uses your psycopg2 helper
# if your project already has a SQLAlchemy session, you can swap to that later

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

def _no_store(resp: HTMLResponse) -> HTMLResponse:
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp
def require_auth(request: Request) -> int | RedirectResponse:
    tid = request.session.get("tenant_id")
    if not tid:
        next_path = request.url.path or "/dashboard/"
        return RedirectResponse(url=f"/login?next={next_path}", status_code=303)
    return int(tid)

def get_tenant_name(tenant_id: int) -> str:
    conn, cur = get_db_connection()
    try:
        cur.execute("SELECT name FROM tenants WHERE id = %s LIMIT 1", (tenant_id,))
        row = cur.fetchone()
        return (row[0] if row else "Unknown").strip() if row and row[0] else "Unknown"
    finally:
        close_db_connection(conn, cur)


@router.get("/", response_class=HTMLResponse, name="dashboard_home")
async def dashboard_home(request: Request):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse): return guard
    tenant_id = guard

    conn, cur = get_db_connection()
    cur.execute("SELECT COUNT(*) FROM customers WHERE tenant_id = %s", (tenant_id,))
    result = cur.fetchone()
    customers_count = result[0] if result is not None else 0

    cur.execute("""
        SELECT COUNT(*) 
        FROM product_variants pv 
        JOIN products p ON p.id = pv.product_id
        WHERE p.tenant_id = %s
    """, (tenant_id,))
    result = cur.fetchone()
    products_count = result[0] if result is not None else 0
    close_db_connection(conn, cur)

    tenant_name = request.session.get("tenant_name") or get_tenant_name(tenant_id)

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "customers_count": customers_count,
            "products_count": products_count,
        },
    )

# ---------- Customers ----------
@router.get("/customer", response_class=HTMLResponse)
async def customers_list(request: Request):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse): return guard
    tenant_id = guard

    conn, cur = get_db_connection()
    cur.execute("""
        SELECT id, name, phone, email, created_at
        FROM customers
        WHERE tenant_id = %s
        ORDER BY created_at DESC
        LIMIT 200
    """, (tenant_id,))
    rows = cur.fetchall()
    close_db_connection(conn, cur)

    customers = [
        {"id": r[0], "name": r[1] or "(no name)", "phone": r[2] or "-", "email": r[3] or "-", "created_at": r[4]}
        for r in rows
    ]
    tenant_name = request.session.get("tenant_name") or get_tenant_name(tenant_id)

    return templates.TemplateResponse(
        "customer_list.html",
        {"request": request, "tenant_id": tenant_id, "tenant_name": tenant_name, "customers": customers},
    )

@router.get("/customer/{customer_id}", response_class=HTMLResponse)
async def customer_detail(request: Request, customer_id: int):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse): return guard
    tenant_id = guard

    conn, cur = get_db_connection()
    cur.execute("""
        SELECT id, name, phone, email, preferred_language, loyalty_points, created_at
        FROM customers
        WHERE id = %s AND tenant_id = %s
        LIMIT 1
    """, (customer_id, tenant_id))
    row = cur.fetchone()
    if not row:
        close_db_connection(conn, cur)
        raise HTTPException(status_code=404, detail="Customer not found")

    customer = {
        "id": row[0], "name": row[1] or "(no name)", "phone": row[2] or "-",
        "email": row[3] or "-", "preferred_language": row[4] or "-",
        "loyalty_points": row[5] or 0, "created_at": row[6]
    }

    # (optional) last 10 orders for this customer
    cur.execute("""
        SELECT o.id, o.order_type, o.status, o.price, o.created_at,
               pv.color, pv.size, pv.fabric, p.name
        FROM orders o
        JOIN product_variants pv ON pv.id = o.product_variant_id
        JOIN products p ON p.id = pv.product_id
        WHERE o.customer_id = %s AND o.tenant_id = %s
        ORDER BY o.created_at DESC
        LIMIT 10
    """, (customer_id, tenant_id))
    orders = cur.fetchall()
    close_db_connection(conn, cur)

    order_items = [
        {
            "id": r[0], "order_type": r[1], "status": r[2], "price": r[3], "created_at": r[4],
            "variant": f"{r[8]} — {r[5]}/{r[6]}/{r[7]}"
        } for r in orders
    ]

    tenant_name = request.session.get("tenant_name") or get_tenant_name(tenant_id)

    return templates.TemplateResponse(
        "customer_detail.html",
        {"request": request, "tenant_id": tenant_id, "tenant_name": tenant_name, "customer": customer, "orders": order_items},
    )

# ---------- Products ----------
@router.get("/product", response_class=HTMLResponse)
async def products_list(request: Request):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse): return guard
    tenant_id = guard

    conn, cur = get_db_connection()
    cur.execute("""
        SELECT pv.id, p.name, pv.color, pv.size, pv.fabric, pv.price, pv.available_stock, pv.is_rental
        FROM product_variants pv
        JOIN products p ON p.id = pv.product_id
        WHERE p.tenant_id = %s
        ORDER BY p.name ASC
        LIMIT 300
    """, (tenant_id,))
    rows = cur.fetchall()
    close_db_connection(conn, cur)

    products = [
        {
            "variant_id": r[0], "product": r[1],
            "attrs": f"{r[2]}/{r[3]}/{r[4]}",
            "price": r[5], "stock": r[6], "is_rental": bool(r[7])
        } for r in rows
    ]

    tenant_name = request.session.get("tenant_name") or get_tenant_name(tenant_id)

    return templates.TemplateResponse(
        "product_list.html",
        {"request": request, "tenant_id": tenant_id, "tenant_name": tenant_name, "products": products},
    )

# --- Chat history page ---
@router.get("/chat/{customer_id}", response_class=HTMLResponse)
async def chat_history(request: Request, customer_id: int):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse): return guard
    tenant_id = guard
    if isinstance(tenant_id, RedirectResponse):
        return tenant_id

    conn, cur = get_db_connection()
    # Get the most recent chat session (or you can remove LIMIT to show all)
    cur.execute("""
        SELECT id, started_at, ended_at, transcript
        FROM chat_sessions
        WHERE customer_id = %s
        ORDER BY started_at DESC
        LIMIT 1
    """, (customer_id,))
    row = cur.fetchone()
    close_db_connection(conn, cur)

    if not row:
        session = None
        messages: List[Dict[str, Any]] = []
    else:
        session = {"id": row[0], "started_at": row[1], "ended_at": row[2]}
        # psycopg2 returns JSON as Python already if column type is json/jsonb.
        # If your driver returns a string, uncomment the json.loads line.
        transcript = row[3]
        # transcript = json.loads(row[3]) if isinstance(row[3], str) else row[3]
        messages = transcript or []

    tenant_name = request.session.get("tenant_name") or get_tenant_name(tenant_id)

    return templates.TemplateResponse(
        "chat_history.html",
        {"request": request, "tenant_id": tenant_id, "tenant_name": tenant_name, "customer_id": customer_id, "session": session, "messages": messages},
    )

# --- Orders page (by customer) ---
@router.get("/orders/{customer_id}", response_class=HTMLResponse)
async def orders_by_customer(request: Request, customer_id: int):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse): return guard
    tenant_id = guard
    if isinstance(tenant_id, RedirectResponse):
        return tenant_id

    conn, cur = get_db_connection()
    cur.execute("""
        SELECT o.id, o.order_type, o.status, o.price, o.created_at,
               pv.color, pv.size, pv.fabric, p.name
        FROM orders o
        JOIN product_variants pv ON pv.id = o.product_variant_id
        JOIN products p ON p.id = pv.product_id
        WHERE o.customer_id = %s AND o.tenant_id = %s
        ORDER BY o.created_at DESC
        LIMIT 200
    """, (customer_id, tenant_id))
    rows = cur.fetchall()
    close_db_connection(conn, cur)

    orders = [
        {
            "id": r[0],
            "order_type": r[1],
            "status": r[2],
            "price": r[3],
            "created_at": r[4],
            "variant": f"{r[8]} — {r[5]}/{r[6]}/{r[7]}",
        }
        for r in (rows or [])
    ]
    tenant_name = request.session.get("tenant_name") or get_tenant_name(tenant_id)

    return templates.TemplateResponse(
        "orders_by_customer.html",
        {"request": request, "tenant_id": tenant_id, "tenant_name": tenant_name, "customer_id": customer_id, "orders": orders},
    )