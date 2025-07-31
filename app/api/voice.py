from fastapi import APIRouter, Request, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from app.db.session import get_db
from app.utils.stt import SarvamSTTStreamHandler
from app.utils.tts import synthesize_text 
from app.core.lang_utils import detect_language
from app.core.ai_reply import generate_reply
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

async def wait_for_user_response(stt, timeout=10.0):
    """Wait for user response after TTS. Exit if silence for too long."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            txt, is_final, lang = await asyncio.wait_for(stt.get_transcript(), timeout=0.5)
            if is_final and txt:
                logging.info(f"👂 Heard after TTS: {txt}")
                return txt
        except asyncio.TimeoutError:
            continue
    logging.info("⌛ No response from user after TTS.")
    return None


@router.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()
    logging.info("✅ WebSocket connection accepted at / stream")

    stt = SarvamSTTStreamHandler()
    stream_sid = None
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
                        txt, is_final,lang= await asyncio.wait_for(stt.get_transcript(), timeout=0.01)
                        if is_final and txt:
                            logging.info(f"🎤 Final transcript: {txt}")
                            
                            lang_code,confidence =await detect_language(txt)
                            logging.info(f"🌐 Detected language: {lang_code}")
                            
                            products = []  # <-- Replace this with your product search logic
                            shop_name = "Krishna Textiles"
                            
                            ai_reply = await generate_reply(
                                user_query=txt,
                                products=products,
                                shop_name=shop_name,
                                action=None,
                                language=lang_code
                            )
                            logging.info(f"🤖 AI Reply: {ai_reply}")
                            
                            audio = await synthesize_text(ai_reply, language_code=lang_code)
                            await speak_pcm(audio, websocket, stream_sid)
                            await stt.reset()

                            # 🕒 Wait for user reply after speaking
                            response_txt = await wait_for_user_response(stt, timeout=30)
                            if response_txt:
                                # You can optionally process this transcript immediately (loop)
                                logging.info("✅ Got response after TTS, will process in next media loop")
                            else:
                                logging.info("🤷‍♂️ No user reply. Waiting for next media stream from client...")
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
