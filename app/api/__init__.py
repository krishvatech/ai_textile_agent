from fastapi import APIRouter
from . import whatsapp, voice, admin

api_router = APIRouter()
api_router.include_router(whatsapp.router, prefix="/whatsapp", tags=["WhatsApp"])
api_router.include_router(voice.router, prefix="/voice", tags=["Voice"])
api_router.include_router(admin.router, prefix="/admin", tags=["Admin"])
