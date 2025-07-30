# app/api/exotel_webhook.py

from fastapi import APIRouter, Request, status
from fastapi.responses import PlainTextResponse
import logging

router = APIRouter()

@router.post("/inbound")
async def handle_inbound_call(request: Request):
    try:
        form_data = await request.form()
        caller = form_data.get("From", "unknown")
        callee = form_data.get("To", "unknown")
        sid = form_data.get("CallSid", "N/A")

        logging.info(f"📞 Inbound Call: From={caller}, To={callee}, SID={sid}")

        # Return TwiML to say something on the call
        response_xml = """
        <Response>
            <Say>Hello! Welcome to Krishva Textile Agent. Your call is connected.</Say>
        </Response>
        """
        return PlainTextResponse(response_xml, media_type="text/xml")

    except Exception as e:
        logging.error(f"❌ Error in inbound call webhook: {e}")