from fastapi import APIRouter, Depends, Request
from fastapi.params import Form
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi.templating import Jinja2Templates

from app.db.models import Tenant
from app.db.session import get_db

router = APIRouter()
templates = Jinja2Templates(directory="app/templates/tenants")

def _no_store(resp: HTMLResponse) -> HTMLResponse:
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@router.get("/login", response_class=HTMLResponse, name="login_page")
async def login_page(request: Request):
    # already logged in? → go to dashboard
    if request.session.get("tenant_id"):
        return RedirectResponse(url="/dashboard", status_code=303)
    resp =templates.TemplateResponse("login.html", {"request": request, "error": None})
    return _no_store(resp)


@router.post("/login", name="login_submit")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Tenant).where(Tenant.email == email))
    tenant = result.scalars().first()

    if not tenant or not tenant.is_active or tenant.password != password:
        resp = templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid email or password"},
            status_code=400,
        )
        return _no_store(resp)

    # success → set session
    request.session["tenant_id"] = tenant.id
    request.session["tenant_name"] = tenant.name
    return RedirectResponse(url="/dashboard", status_code=303)


@router.post("/logout")
async def logout(request: Request):
    request.session.clear()
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("txa_session")
    return _no_store(resp)