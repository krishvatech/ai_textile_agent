from fastapi import APIRouter, Request, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from app.db.session import get_db
from app.utils.stt import SarvamSTTStreamHandler
from app.utils.tts import synthesize_text 
from app.core.lang_utils import detect_language
from app.core.intent_utils import detect_textile_intent_openai
from app.core.ai_reply import TextileAnalyzer
from sqlalchemy import text
from fastapi import Depends
import json
import asyncio
import logging
import base64
import time
import re
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

import re

async def normalize_size(transcript: str) -> str | None:
    text = transcript.lower().strip().replace('.', '')

    # English patterns (your existing ones)
    if re.search(r'double\s*(x{1,3}l|ex|excel)', text) or re.search(r'extra\s*extra\s*large', text):
        return "XXL"
    if re.search(r'extra\s*large|excel', text):
        return "XL"
    if re.search(r'large', text):
        return "L"
    if re.search(r'medium|med', text):
        return "M"
    if re.search(r'small|sm', text):
        return "S"
    if re.search(r'extra\s*small|xs', text):
        return "XS"
    if re.search(r'double\s*extra\s*small|xxs', text):
        return "XXS"

    # Gujarati size words (Unicode strings)
    # 'એક્સેલ' = XL, 'ડબલ એક્સેલ' = XXL, 'લાર્જ' = L, 'મીડિયમ' = M, 'સ્મોલ' = S, 'એક્સએસ' = XS, 'ડબલ એક્સએસ' = XXS
    if re.search(r'ડબલ\s*એક્સેલ|ડબલએક્સેલ', text):
        return "XXL"
    if re.search(r'ડબલ\s*એક્સલ|ડબલએક્સલ.', text):
        return "XXL"
    if re.search(r'એક્સેલ', text):
        return "XL"
    if re.search(r'એક્સલ', text):
        return "XL"
    if re.search(r'લાર્જ', text):
        return "L"
    if re.search(r'મીડિયમ', text):
        return "M"
    if re.search(r'સ્મોલ', text):
        return "S"
    if re.search(r'એક્સએસ', text):
        return "XS"
    if re.search(r'ડબલ\s*એક્સએસ|ડબલએક્સએસ', text):
        return "XXS"

    # Hindi size words (Unicode strings)
    # 'डबल एक्सेल' = XXL, 'एक्सेल' = XL, 'लार्ज' = L, 'मीडियम' = M, 'स्माल' = S, 'एक्सएस' = XS, 'डबल एक्सएस' = XXS
    if re.search(r'डबल\s*एक्सेल', text):
        return "XXL"
    if re.search(r'एक्सेल', text):
        return "XL"
    if re.search(r'लार्ज', text):
        return "L"
    if re.search(r'मीडियम', text):
        return "M"
    if re.search(r'स्माल', text):
        return "S"
    if re.search(r'एक्सएस', text):
        return "XS"
    if re.search(r'डबल\s*एक्सएस', text):
        return "XXS"

    # Also handle exact size code inputs like 'xl', 'xxl', etc.
    sizes = ['xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxxl']
    if text in sizes:
        return text.upper()

    return None


@router.websocket("/stream")
async def stream_audio(websocket: WebSocket,db=Depends(get_db)):
    logging.info("Incoming WebSocket connection - before accept")
    await websocket.accept()
    logging.info("✅ WebSocket connection accepted at /stream")
    analyzer.reset()  # <-- you need to add this method inside TextileAnalyzer
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
        last_user_lang = lang_code
        while True:
            try:
                txt, is_final, lang = await asyncio.wait_for(stt.get_transcript(), timeout=0.2)
                if is_final and txt:
                    logging.info(f"Final transcript: {txt}")
                    if re.fullmatch(r'[\W_]+', txt.strip()):
                        logging.info("Transcript only punctuation, ignoring")
                        continue

                    lang,_ = await detect_language(txt,last_user_lang)
                    normalized_size=await normalize_size(txt)
                    intent, new_entities, intent_confidence = await detect_textile_intent_openai(txt, lang)
                    if intent_confidence < 0.5:
                        logging.info(f"Ignoring transcript due to low intent confidence: {intent_confidence}")
                        continue  # Skip processing
                    if normalized_size:
                        new_entities['size'] = normalized_size
                        logging.info(f"Overriding detected size with normalized size: {normalized_size}")
                    last_activity = time.time()  # Reset silence timer
                    ai_reply = await analyzer.analyze_message(
                        text=txt,
                        tenant_id=tenant_id ,
                        language=lang,
                        intent=intent,
                        new_entities=new_entities,         # correct keyword
                        intent_confidence=intent_confidence # correct keyword
                    )
                    logging.info(f"🤖 AI Reply In Dictionary : {ai_reply}")
                    answer_text = ai_reply.get('answer', '')

                    logging.info(f"AI Reply: {answer_text}")
                    audio = await synthesize_text(answer_text, language_code=lang)
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
