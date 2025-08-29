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
    if isinstance(guard, RedirectResponse): 
        return guard
    tenant_id = guard

    conn, cur = get_db_connection()

    # --- KPIs ---
    # Customers
    cur.execute("SELECT COUNT(*) FROM customers WHERE tenant_id = %s", (tenant_id,))
    customers_count = (cur.fetchone() or [0])[0]

    # Products (product-level, not variants)
    cur.execute("SELECT COUNT(*) FROM products WHERE tenant_id = %s", (tenant_id,))
    products_count = (cur.fetchone() or [0])[0]

    # Products that have at least one rental variant
    cur.execute("""
        SELECT COUNT(DISTINCT p.id)
        FROM products p
        JOIN product_variants pv ON pv.product_id = p.id
        WHERE p.tenant_id = %s AND COALESCE(pv.is_rental,false) = true
    """, (tenant_id,))
    rental_products_count = (cur.fetchone() or [0])[0]

    # Products that have at least one selling (non-rental) variant
    cur.execute("""
        SELECT COUNT(DISTINCT p.id)
        FROM products p
        JOIN product_variants pv ON pv.product_id = p.id
        WHERE p.tenant_id = %s AND COALESCE(pv.is_rental,false) = false
    """, (tenant_id,))
    selling_products_count = (cur.fetchone() or [0])[0]

    # Orders (selling) and Rentals (separate table)
    cur.execute("SELECT COUNT(*) FROM orders   WHERE tenant_id = %s", (tenant_id,))
    orders_count = (cur.fetchone() or [0])[0]

    cur.execute("SELECT COUNT(*) FROM rentals  WHERE tenant_id = %s", (tenant_id,))
    rentals_count = (cur.fetchone() or [0])[0]

    # --- Charts ---
    # Monthly series for selling orders
    cur.execute("""
        SELECT DATE_TRUNC('month', start_date) AS mth, COUNT(*)
        FROM orders
        WHERE tenant_id = %s
        GROUP BY mth
        ORDER BY mth
    """, (tenant_id,))
    sell_rows = cur.fetchall() or []

    # Monthly series for rentals
    cur.execute("""
        SELECT DATE_TRUNC('month', rental_start_date) AS mth, COUNT(*)
        FROM rentals
        WHERE tenant_id = %s
        GROUP BY mth
        ORDER BY mth
    """, (tenant_id,))
    rent_rows = cur.fetchall() or []

    # Top 5 by selling orders (count)
    cur.execute("""
        SELECT p.name, COUNT(*) AS cnt
        FROM orders o
        JOIN product_variants pv ON pv.id = o.product_variant_id
        JOIN products p ON p.id = pv.product_id
        WHERE o.tenant_id = %s
        GROUP BY p.name
        ORDER BY cnt DESC
        LIMIT 5
    """, (tenant_id,))
    top_sell_rows = cur.fetchall() or []

    # Top 5 by rentals (count)
    cur.execute("""
        SELECT p.name, COUNT(*) AS cnt
        FROM rentals r
        JOIN product_variants pv ON pv.id = r.product_variant_id
        JOIN products p ON p.id = pv.product_id
        WHERE r.tenant_id = %s
        GROUP BY p.name
        ORDER BY cnt DESC
        LIMIT 5
    """, (tenant_id,))
    top_rent_rows = cur.fetchall() or []

    close_db_connection(conn, cur)

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

    # Merge top lists → same set of product labels with both counts
    from collections import defaultdict
    top_map = defaultdict(lambda: {"sell": 0, "rent": 0})
    for name, cnt in top_sell_rows:
        top_map[name]["sell"] = cnt
    for name, cnt in top_rent_rows:
        top_map[name]["rent"] = cnt

    # Sort by combined count desc and take top 5
    merged_top = sorted(
        top_map.items(),
        key=lambda kv: (kv[1]["sell"] + kv[1]["rent"]),
        reverse=True
    )[:5]

    top_labels = [name for name, _ in merged_top]
    top_sell   = [vals["sell"] for _, vals in merged_top]
    top_rent   = [vals["rent"] for _, vals in merged_top]

    tenant_name = request.session.get("tenant_name") or get_tenant_name(tenant_id)

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,

            # KPI cards
            "customers_count": customers_count,
            "products_count": products_count,
            "rental_products_count": rental_products_count,
            "selling_products_count": selling_products_count,
            "orders_count": orders_count,     # selling orders
            "rentals_count": rentals_count,   # rentals table

            # Charts (already JSON-serializable)
            "chart_labels": labels,
            "chart_sell": sell_series,
            "chart_rent": rent_series,

            "top_labels": top_labels,
            "top_sell": top_sell,
            "top_rent": top_rent,
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
        SELECT id, whatsapp_id, name, preferred_language, phone, email, created_at, is_active, loyalty_points
        FROM customers
        WHERE tenant_id = %s
        ORDER BY created_at DESC
        LIMIT 200
    """, (tenant_id,))
    rows = cur.fetchall()
    close_db_connection(conn, cur)

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

    # filter on the LIST page
    t = (request.query_params.get("type") or "").lower()
    if t not in ("", "all", "buy", "rent"):
        t = ""
    current_type = "all" if t in ("", "all") else t

    conn, cur = get_db_connection()
    try:
        having = ""
        if current_type == "buy":
            having = "HAVING COALESCE(SUM(CASE WHEN pv.is_rental = FALSE THEN 1 END), 0) > 0"
        elif current_type == "rent":
            having = "HAVING COALESCE(SUM(CASE WHEN pv.is_rental = TRUE  THEN 1 END), 0) > 0"

        q = f"""
            SELECT p.id,
                   p.name,
                   COUNT(pv.id) AS variant_count,
                   COALESCE(SUM(CASE WHEN pv.is_rental THEN 1 ELSE 0 END), 0)  AS rental_count,
                   COALESCE(SUM(CASE WHEN NOT pv.is_rental THEN 1 ELSE 0 END), 0) AS buy_count
            FROM products p
            LEFT JOIN product_variants pv ON pv.product_id = p.id
            WHERE p.tenant_id = %s
            GROUP BY p.id, p.name
            {having}
            ORDER BY p.name ASC
            LIMIT 300
        """
        cur.execute(q, (tenant_id,))
        rows = cur.fetchall()
    finally:
        close_db_connection(conn, cur)

    products = [
        {
            "product_id": r[0],
            "product": r[1],
            "variant_count": r[2] or 0,
            "rental_count": r[3] or 0,
            "buy_count": r[4] or 0,
        } for r in rows
    ]

    tenant_name = request.session.get("tenant_name") or get_tenant_name(tenant_id)
    return templates.TemplateResponse(
        "product_list.html",
        {
            "request": request,
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "products": products,
            "current_type": current_type,   # <-- for tabs highlight
        },
    )

@router.get("/product/{product_id}", response_class=HTMLResponse, name="product_detail")
async def product_detail(request: Request, product_id: int):
    guard = require_auth(request)
    if isinstance(guard, RedirectResponse):
        return guard
    tenant_id = guard

    # read filter
    t = (request.query_params.get("type") or "").lower()
    if t not in ("", "all", "buy", "rent"):
        t = ""
    current_type = "all" if t in ("", "all") else t

    conn, cur = get_db_connection()
    try:
        # Ensure the product belongs to this tenant and fetch product name
        cur.execute("""
            SELECT id, name
            FROM products
            WHERE id = %s AND tenant_id = %s
            LIMIT 1
        """, (product_id, tenant_id))
        prod = cur.fetchone()
        if not prod:
            raise HTTPException(status_code=404, detail="Product not found")

        product = {"id": prod[0], "name": prod[1]}

        where = "WHERE pv.product_id = %s"
        params = [product_id]
        if current_type == "rent":
            where += " AND pv.is_rental = TRUE"
        elif current_type == "buy":
            where += " AND pv.is_rental = FALSE"

        # Get all variants for this product
        cur.execute(f"""
            SELECT pv.id, pv.color, pv.size, pv.fabric,
                   pv.price, pv.rental_price, pv.available_stock, pv.is_rental
            FROM product_variants pv
            {where}
            ORDER BY pv.color, pv.size, pv.fabric
        """, tuple(params))
        # cur.execute("""
        #     SELECT id, color, size, fabric, price, available_stock, is_rental
        #     FROM product_variants
        #     WHERE product_id = %s
        #     ORDER BY color, size, fabric
        # """, (product_id,))
        rows = cur.fetchall()
    finally:
        close_db_connection(conn, cur)

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

    tenant_name = request.session.get("tenant_name") or get_tenant_name(tenant_id)
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