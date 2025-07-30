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

        return PlainTextResponse("OK", status_code=status.HTTP_200_OK)

    except Exception as e:
        logging.error(f"❌ Error in inbound call webhook: {e}")
        return PlainTextResponse("Internal Server Error", status_code=500)
