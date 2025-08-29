from fastapi import APIRouter, Request                 # ⟵ add Request here
from fastapi.responses import RedirectResponse, JSONResponse  # ⟵ add thi
from . import whatsapp, voice, admin, login, dashboard
from .exotel_webhook import router as stream_server_router

api_router = APIRouter()
api_router.include_router(stream_server_router,prefix="/voice")
api_router.include_router(whatsapp.router, prefix="/whatsapp", tags=["WhatsApp"]) 
api_router.include_router(voice.router, prefix="/voice", tags=["Voice"])
api_router.include_router(admin.router, prefix="/admin", tags=["Admin"])
api_router.include_router(login.router, tags=["Login"])
api_router.include_router(dashboard.router, prefix="/dashboard")

@api_router.get("/", include_in_schema=False)
async def home(request: Request):
    if request.session.get("tenant_id"):
        return RedirectResponse("/dashboard", status_code=303)
    return RedirectResponse("/login", status_code=303)

# (optional) move your health JSON off "/" if you had one
@api_router.get("/healthz", tags=["Health"])
def health():
    return {"message": "✅ API is live"}