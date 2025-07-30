# app/api/exotel_webhook.py

from fastapi import APIRouter, Request, status
from fastapi.responses import PlainTextResponse
import logging

router = APIRouter()

@router.post("/inbound")
async def handle_inbound_call(request: Request):
    try:
        print("🔔 Incoming call webhook received!")
        response_xml = """
        <Response>
            <Say>Hello! This is a test from Krishva Agent. If you hear this, everything works.</Say>
        </Response>
        """
        return PlainTextResponse(response_xml, media_type="text/xml")

    except Exception as e:
        logging.error(f"❌ Error in inbound call webhook: {e}")