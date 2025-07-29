from fastapi import APIRouter

api_router = APIRouter()

# Import and include all channel routers
from . import whatsapp
api_router.include_router(whatsapp.router, prefix="/whatsapp", tags=["WhatsApp"])

# (Future) from . import voice
# api_router.include_router(voice.router, prefix="/voice", tags=["Voice"])
