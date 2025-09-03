# app/api/login.py
from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from fastapi.templating import Jinja2Templates

# 🔁 adjust this import to your project path (e.g., app.db.session.get_db)
from app.db.session import get_db

router = APIRouter()
templates = Jinja2Templates(directory="app/templates/tenants")

def _no_store(resp: HTMLResponse) -> HTMLResponse:
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@router.get("/login", response_class=HTMLResponse, name="login_page")
async def login_page(request: Request, next: str | None = None):
    role = request.session.get("role")
    if role == "superadmin":
        return RedirectResponse(url="/admin", status_code=303)
    if role:
        return RedirectResponse(url="/dashboard", status_code=303)
    resp = templates.TemplateResponse("login.html", {"request": request, "error": None, "next": next})
    return _no_store(resp)

@router.post("/login", name="login_submit")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    # Look up user in public.users (password stored in "hashed_password" as plain text)
    q = text("""
        SELECT id, email, hashed_password, role::text, tenant_id, is_active
        FROM public.users
        WHERE email = :email
        LIMIT 1
    """)
    row = (await db.execute(q, {"email": email.strip()})).fetchone()

    if not row:
        resp = templates.TemplateResponse("login.html", {"request": request, "error": "Invalid email or password"})
        return _no_store(resp)

    user_id, _email, stored_pw, role, tenant_id, is_active = row

    # 🔐 Plain-text check (as requested)
    if (not is_active) or (password.strip() != (stored_pw or "")):
        resp = templates.TemplateResponse("login.html", {"request": request, "error": "Invalid email or password"})
        return _no_store(resp)

    # Success — set session
    request.session.clear()
    request.session.update({
        "user_id": int(user_id),
        "email": _email,
        "role": role,  # 'superadmin' | 'tenant_admin' | ...
        "tenant_id": int(tenant_id) if tenant_id is not None else None,
    })

    # Redirect by role (or to `next`)
    if next:
        dest = next
    elif role == "superadmin":
        dest = "/admin"
    else:
        dest = "/dashboard"

    return RedirectResponse(url=dest, status_code=303)

@router.post("/logout")
async def logout(request: Request):
    request.session.clear()
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("txa_session")
    return _no_store(resp)
