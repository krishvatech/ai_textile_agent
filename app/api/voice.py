from fastapi import APIRouter, Request, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect,Response
from app.db.session import get_db
from app.utils.stt import SarvamSTTStreamHandler
from app.utils.tts import synthesize_text 
from app.core.lang_utils import detect_language
from app.core.intent_utils import detect_textile_intent_openai
from app.core.ai_reply import TextileAnalyzer
from sqlalchemy import text
from fastapi import Depends
from datetime import datetime
import json
import asyncio
import logging
import base64
import time
import re
import os

router = APIRouter()
logging.basicConfig(level=logging.INFO)

      
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

async def new_stt_stream() -> SarvamSTTStreamHandler:
    stt = SarvamSTTStreamHandler()
    await stt.start_stream()
    return stt

@router.websocket("/stream")
async def stream_audio(websocket: WebSocket,db=Depends(get_db)):
    print("Incoming WebSocket connection - before accept")
    logging.info("Incoming WebSocket connection - before accept")
    await websocket.accept()
    logging.info("✅ WebSocket connection accepted at /stream")
    print("✅ WebSocket connection accepted at /stream")
    analyzer.reset()  # <-- you need to add this method inside TextileAnalyzer
    stt = await new_stt_stream()
    stream_sid = None
    lang_code = 'en-IN'  # Default language
    bot_is_speaking = False
    tts_task = None
    current_language = None
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
        nonlocal stream_sid, tenant_id, db
        is_outbound_call = None  # Track call type
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                event_type = message.get("event")
                if bot_is_speaking:
                    await stop_tts()
                
                
                if event_type == "connected":
                    greeting = "Hello..I am From Krishvatech Texttile"
                    audio = await synthesize_text(greeting, 'en-IN')
                    await speak_pcm(audio, websocket, stream_sid)
                    
                if event_type == "start":
                    stream_sid = message.get("stream_sid")
                    start_payload = message.get("start", {})
                    logging.info(f"start_payload : {start_payload}")
                    phone_number = start_payload.get("to", "unknown")
                    logging.info(f"Final phone: {phone_number}")
                    
                    if phone_number:
                        tenant_id = await get_tenant_id_by_phone(phone_number, db)
                    logging.info(f"Tenant ID resolved: {tenant_id}")
                    logging.info(f"stream_sid: {stream_sid}")
                    custom_params = start_payload.get("custom_parameters", {})
                    print("custom_params=",custom_params)
                    is_outbound = False
                    if custom_params:
                        key = list(custom_params.keys())[0]
                        parts = key.split("|")
                        if parts:
                            call_type = parts[0]  # e.g., 'outbound'
                            is_outbound = call_type == "outbound"  # ✅ set flag properly

                        if len(parts) > 1:
                            database = parts[1] 
                        if len(parts) > 2:
                            call_session_id = parts[2]  # ✅ Set it here!
                if event_type == "media":
                    pcm = base64.b64decode(message["media"]["payload"])
                    await stt.send_audio_chunk(pcm)  # Feed chunk to STT (non-blocking)
                if event_type == "stop":
                    logging.info("Call ended.")
                    break
        except WebSocketDisconnect:
            logging.info(":octagonal_sign: WebSocket disconnected")
            
    async def process_transcripts():
        """Process STT transcripts, generate replies, and handle TTS"""
        nonlocal current_language
        last_activity = time.time()
        last_user_lang = lang_code
        while True:
            try:
                txt, is_final, lang = await asyncio.wait_for(stt.get_transcript(), timeout=0.2)
                if is_final and txt:
                    logging.info(f"Final transcript: {txt}")
                    if re.fullmatch(r'[\W_]+', txt.strip()):
                        logging.info("Transcript only punctuation, ignoring")
                        continue
                    start_lang = time.perf_counter()
                    detected_lang,_ = await detect_language(txt, last_user_lang)

                    # Fix conversation language once set, but update if neutral or English greetings only
                    if current_language is None:
                        current_language = detected_lang
                        logging.info(f"Conversation language set to {current_language}")
                    else:
                        # If current_language is neutral or en-IN (greeting), update to detected_lang if meaningful
                        if current_language in ['neutral', 'en-IN'] and detected_lang in ['hi-IN', 'gu-IN']:
                            current_language = detected_lang
                            logging.info(f"Conversation language updated to {current_language}")
                        else:
                            logging.info(f"Conversation language remains as {current_language}")

                    lang = current_language
                    last_user_lang = current_language
                    
                    start_intent = time.perf_counter()
                    intent, new_entities, intent_confidence = await detect_textile_intent_openai(txt, lang)
                    elapsed_intent = (time.perf_counter() - start_intent) * 1000
                    logging.info(f"detect_textile_intent_openai took {elapsed_intent:.2f} ms")
                    
                    if intent_confidence < 0.5:
                        logging.info(f"Ignoring transcript due to low intent confidence: {intent_confidence}")
                        continue  # Skip processing
                    last_activity = time.time()  # Reset silence timer
                    
                    start_analyzer = time.perf_counter()
                    ai_reply = await analyzer.analyze_message(
                        text=txt,
                        tenant_id=tenant_id ,
                        language=lang,
                        intent=intent,
                        new_entities=new_entities,         # correct keyword
                        intent_confidence=intent_confidence # correct keyword
                    )
                    elapsed_analyzer = (time.perf_counter() - start_analyzer) * 1000
                    logging.info(f"analyzer.analyze_message took {elapsed_analyzer:.2f} ms")

                    last_activity = time.time()
                    
                    logging.info(f"🤖 AI Reply In Dictionary : {ai_reply}")
                    answer_text = ai_reply.get('answer', '')
                    
                    if lang == "neutral":
                        tts_lang = 'en-IN'  # or set your preferred default language
                    else:
                        tts_lang = lang
                    audio = await synthesize_text(answer_text, language_code=tts_lang)
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
