from fastapi import APIRouter
from . import whatsapp, voice, admin
import app.utils.whatsapp_utils as whatsapp_utils
from .exotel_webhook import router as stream_server_router

api_router = APIRouter()
api_router.include_router(stream_server_router,prefix="/voice")
api_router.include_router(whatsapp_utils.router, prefix="/whatsapp", tags=["WhatsApp"]) 
api_router.include_router(voice.router, prefix="/voice", tags=["Voice"])
api_router.include_router(admin.router, prefix="/admin", tags=["Admin"])