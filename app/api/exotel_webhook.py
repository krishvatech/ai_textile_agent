# app/api/exotel_webhook.py

from fastapi import APIRouter, Request, status,WebSocketDisconnect,WebSocket
from fastapi.responses import PlainTextResponse

import logging

router = APIRouter()


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