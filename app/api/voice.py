from fastapi import APIRouter, Request, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from app.db.session import get_db
from app.utils.stt import SarvamSTTStreamHandler
from app.utils.tts import synthesize_text 
from app.core.ai_reply import TextileAnalyzer
from sqlalchemy import text
from fastapi import Depends
import json
import asyncio
import logging
import base64
import time
import asyncio

router = APIRouter()

user_context = {}
analyzer = TextileAnalyzer()

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
        await asyncio.sleep(0.05)

async def get_tenant_id_by_phone(phone_number: str, db):
    query = text("SELECT id FROM tenants WHERE phone_number = :phone AND is_active = true LIMIT 1")
    result = await db.execute(query, {"phone": phone_number})
    row = result.fetchone()
    if row:
        return row[0]
    return None

@router.websocket("/stream")
async def stream_audio(websocket: WebSocket,db=Depends(get_db)):
    logging.info("Incoming WebSocket connection - before accept")
    await websocket.accept()
    logging.info("✅ WebSocket connection accepted at /stream")

    stt = SarvamSTTStreamHandler()
    stream_sid = None
    lang_code = 'en-IN'  # Default language
    bot_is_speaking = False
    tts_task = None
    tenant_id = None
    
    async def stop_tts():
        """Stop TTS if bot is speaking"""
        nonlocal bot_is_speaking, tts_task
        if bot_is_speaking:
            logging.info("Interrupt detected, stopping TTS.")
            bot_is_speaking = False
            if tts_task:
                tts_task.cancel()  # Cancel the ongoing TTS task
                await tts_task  #
    async def receive_messages():
        """Receive WebSocket messages and feed audio to STT"""
        nonlocal stream_sid
        nonlocal tenant_id
        nonlocal db
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                event_type = message.get("event")
                if bot_is_speaking:
                    await stop_tts()
                
                if event_type == "connected":
                    greeting = "How can I help you today?"
                    audio = await synthesize_text(greeting, lang_code)
                    await speak_pcm(audio, websocket, stream_sid)
                    
                elif event_type == "start":
                    stream_sid = message.get("stream_sid")
                    start_payload = message.get("start", {})
                    logging.info(f"start_payload : {start_payload}")
                    phone_number = start_payload.get("to", "unknown")
                    logging.info(f"Final phone: {phone_number}")
                    if phone_number:
                        tenant_id = await get_tenant_id_by_phone(phone_number, db)
                    logging.info(f"Tenant ID resolved: {tenant_id}")
                    logging.info(f"stream_sid: {stream_sid}")
                    
                elif event_type == "media":
                    pcm = base64.b64decode(message["media"]["payload"])
                    await stt.send_audio_chunk(pcm)  # Feed chunk to STT (non-blocking)
                elif event_type == "stop":
                    logging.info("Call ended.")
                    break
        except WebSocketDisconnect:
            logging.info(":octagonal_sign: WebSocket disconnected")
            
    async def process_transcripts():
        """Process STT transcripts, generate replies, and handle TTS"""
        last_activity = time.time()
        while True:
            try:
                txt, is_final, lang = await asyncio.wait_for(stt.get_transcript(), timeout=0.2)
                if is_final and txt:
                    logging.info(f"Final transcript: {txt}")
                    last_activity = time.time()  # Reset silence timer
                    ai_reply = await analyzer.analyze_message(
                        text=txt,
                        tenant_id=tenant_id ,
                    )
                    answer_text = ai_reply.get('answer', '')

                    logging.info(f"AI Reply: {answer_text}")
                    audio = await synthesize_text(answer_text, language_code=lang_code)
                    await speak_pcm(audio, websocket, stream_sid)

                elif txt:
                    logging.info(f"Interim: {txt}")
                    last_activity = time.time()
            except asyncio.TimeoutError:
                # Check for prolonged silence (e.g., no response after TTS)
                if time.time() - last_activity > 15:  # Adjustable timeout
                    logging.info("No response from user after TTS.")
                    last_activity = time.time()  # Reset or handle end of turn
                continue

    try:
        # Start both tasks concurrently
        receiver_task = asyncio.create_task(receive_messages())
        processor_task = asyncio.create_task(process_transcripts())
        # Wait for either task to complete (e.g., disconnect or stop)
        done, pending = await asyncio.wait(
            [receiver_task, processor_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
    except Exception as e:
        logging.error(f"Error in WebSocket: {e}")
    finally:
        await stt.close_stream()
