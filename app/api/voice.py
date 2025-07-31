from fastapi import APIRouter, Request, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from app.db.session import get_db
from app.utils.stt import SarvamSTTStreamHandler
from app.utils.tts import synthesize_text 
from core.lang_utils import detect_language
import json
import asyncio
import logging
import base64
import time
import asyncio


router = APIRouter()

async def speak_pcm(pcm_audio: bytes, websocket: WebSocket, stream_sid: str):
    """Send TTS audio in chunks over WebSocket"""
    chunk_size = 3200
    for i in range(0, len(pcm_audio), chunk_size):
        chunk = pcm_audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk += b'\0' * (chunk_size - len(chunk))
        b64_chunk = base64.b64encode(chunk).decode("ascii")
        await websocket.send_json({
            'event': 'media',
            'stream_sid': stream_sid,
            'media': {'payload': b64_chunk}
        })
        await asyncio.sleep(0.2)


@router.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()
    logging.info("✅ WebSocket connection accepted at /stream")

    stt = SarvamSTTStreamHandler()
    stream_sid = None
    bot_is_speaking = False
    last_user_input_time = time.time()
    lang_code = 'en-IN'

    try:
        while True:
            message = await websocket.receive_text()
            message = json.loads(message)
            event_type = message.get("event")

            if event_type == "connected":
                greeting = "How can I help you today?"
                audio = await synthesize_text(greeting, lang_code)
                await speak_pcm(audio, websocket, stream_sid)

            elif event_type == "start":
                stream_sid = message.get("stream_sid")
                logging.info(f"🔁 stream_sid: {stream_sid}")

            elif event_type == "media":
                pcm = base64.b64decode(message["media"]["payload"])
                await stt.send_audio_chunk(pcm)
                try:
                    while True:
                        txt, is_final, lang = await asyncio.wait_for(stt.get_transcript(), timeout=0.01)
                        if is_final and txt:
                            logging.info(f"🎤 Final transcript: {txt}")
                            lang_code=detect_language(txt)
                            logging.info(f" lang_code: {lang_code}")
                            response = f"You said: {txt}"  # Replace with AI logic if needed
                            audio = await synthesize_text(response, lang_code)
                            await speak_pcm(audio, websocket, stream_sid)
                            await stt.reset()
                        elif txt:
                            logging.info(f"📝 Interim: {txt}")
                except asyncio.TimeoutError:
                    pass

            elif event_type == "stop":
                logging.info("🛑 Call ended.")
                break

    except WebSocketDisconnect:
        logging.info("🛑 WebSocket disconnected")
    finally:
        await stt.close_stream()
