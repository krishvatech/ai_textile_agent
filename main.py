from fastapi import FastAPI
from app.api import api_router,stream_server
from fastapi import WebSocket


app = FastAPI(title="Universal AI Textile Agent")
app.include_router(api_router)

@app.websocket("/stream")
async def websocket_route(websocket: WebSocket):
    await stream_server.stream_handler(websocket)