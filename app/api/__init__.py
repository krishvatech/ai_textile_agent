from fastapi import APIRouter
from . import whatsapp, voice, admin
from .exotel_webhook import router as stream_server_router
from . import outbound_call 

api_router = APIRouter()
api_router.include_router(stream_server_router)
api_router.include_router(whatsapp.router, prefix="/whatsapp", tags=["WhatsApp"])
api_router.include_router(voice.router, prefix="/voice", tags=["Voice"])
api_router.include_router(admin.router, prefix="/admin", tags=["Admin"])
api_router.include_router(outbound_call.router, prefix="/outbound-call", tags=["Outbound Call"])
