# app/api/exotel_webhook.py

from fastapi import APIRouter, Request, status,WebSocketDisconnect,WebSocket
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

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection accepted")

    try:
        while True:
            data = await websocket.receive_text()
            logging.info(f"Received: {data}")
            await websocket.send_text(f"Echo: {data}")

    except WebSocketDisconnect:
        logging.info("Client disconnected")