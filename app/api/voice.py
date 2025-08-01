from fastapi import APIRouter, Request, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from app.db.session import get_db
from app.utils.stt import SarvamSTTStreamHandler
from app.utils.tts import synthesize_text 
from app.core.lang_utils import detect_language
from app.core.ai_reply import generate_reply
from app.core.intent_utils import detect_textile_intent_openai
from app.utils.pinecone_utils import query_products
from app.core.attribute_extraction import extract_dynamic_attributes
import json
import asyncio
import logging
import base64
import time
import asyncio

router = APIRouter()

user_context = {}

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

# @router.websocket("/stream")
# async def stream_audio(websocket: WebSocket):
#     await websocket.accept()
#     logging.info("✅ WebSocket connection accepted at / stream")

#     stt = SarvamSTTStreamHandler()
#     stream_sid = None
#     lang_code = 'en-IN'
#     bot_is_speaking = False
    
#     async def stop_tts():
#         """Stop TTS if bot is speaking"""
#         nonlocal bot_is_speaking
#         if bot_is_speaking:
#             logging.info(":red_circle: Interrupt detected, stopping TTS.")
#             bot_is_speaking = False
#             # Here, you can stop or

#     async def receive_messages():
#         """Task to receive WebSocket messages and feed audio to STT"""
#         nonlocal stream_sid
#         try:
#             while True:
#                 message = await websocket.receive_text()
#                 if bot_is_speaking:
#                     await stop_tts()
                    
#                 message = json.loads(message)
#                 event_type = message.get("event")
#                 if event_type == "connected":
#                     greeting = "How can I help you today?"
#                     audio = await synthesize_text(greeting, lang_code)
#                     await speak_pcm(audio, websocket, stream_sid)
                    
#                 elif event_type == "start":
#                     stream_sid = message.get("stream_sid")
#                     logging.info(f"stream_sid: {stream_sid}")
                    
#                 elif event_type == "media":
#                     pcm = base64.b64decode(message["media"]["payload"])
#                     await stt.send_audio_chunk(pcm)  # Feed chunk to STT (non-blocking)
#                 elif event_type == "stop":
#                     logging.info("Call ended.")
#                     break
#         except WebSocketDisconnect:
#             logging.info(":octagonal_sign: WebSocket disconnected")
            
#     async def process_transcripts():
#         """Task to process STT transcripts, generate replies, and handle TTS"""
#         last_activity = time.time()
#         while True:
#             try:
#                 txt, is_final, lang = await asyncio.wait_for(stt.get_transcript(), timeout=0.2)
#                 if is_final and txt:
#                     logging.info(f"Final transcript: {txt}")
#                     last_activity = time.time()  # Reset silence timer
#                     extracted_attributes = extract_dynamic_attributes(txt)
#                     logging.info(f"Extracted attributes: {extracted_attributes}")

#                     # Update the user context with extracted attributes
#                     if 'color' in extracted_attributes:
#                         user_context['color'] = extracted_attributes['color']
#                     if 'fabric' in extracted_attributes:
#                         user_context['fabric'] = extracted_attributes['fabric']
                        
#                     print(f"Here are some {user_context['color']}")
#                     print(f"Here are some {user_context['fabric']}")
                    
#                     lang_code, confidence = await detect_language(txt)
#                     user_context[stream_sid] = {'language': lang_code}
#                     intent, filtered_entities, intent_confidence = await detect_textile_intent_openai(txt, lang_code)
#                     logging.info(f"Detected language: {lang_code}")
#                     logging.info(f"Detected Intent: {intent}")
#                     products = []
#                     shop_name = "Krishna Textiles"
#                     if intent == "product_search":
#                         products = await query_products(txt, lang=lang_code)
#                     print("Products=",products)
#                     ai_reply = await generate_reply(
#                         user_query=txt,
#                         products=products,
#                         shop_name=shop_name,
#                         action=None,
#                         language=lang_code,
#                         intent=intent,
#                     )
                    
#                     logging.info(f"AI Reply: {ai_reply}")
#                     audio = await synthesize_text(ai_reply, language_code=lang_code)
#                     await speak_pcm(audio, websocket, stream_sid)

#                 elif txt:
#                     logging.info(f"Interim: {txt}")
#                     last_activity = time.time()
#             except asyncio.TimeoutError:
#                 # Check for prolonged silence (e.g., no response after TTS)
#                 if time.time() - last_activity > 15:  # Adjustable timeout
#                     logging.info("No response from user after TTS.")
#                     last_activity = time.time()  # Reset or handle end of turn
#                 continue
            
#     try:
#         # Start both tasks concurrently
#         receiver_task = asyncio.create_task(receive_messages())
#         processor_task = asyncio.create_task(process_transcripts())
#         # Wait for either task to complete (e.g., disconnect or stop)
#         done, pending = await asyncio.wait(
#             [receiver_task, processor_task],
#             return_when=asyncio.FIRST_COMPLETED
#         )
#         for task in pending:
#             task.cancel()
#     except Exception as e:
#         logging.error(f"Error in WebSocket: {e}")
#     finally:
#         await stt.close_stream()

@router.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()
    logging.info("✅ WebSocket connection accepted at /stream")

    stt = SarvamSTTStreamHandler()
    stream_sid = None
    lang_code = 'en-IN'  # Default language
    bot_is_speaking = False
    
    async def stop_tts():
        """Stop TTS if bot is speaking"""
        nonlocal bot_is_speaking
        if bot_is_speaking:
            logging.info(":red_circle: Interrupt detected, stopping TTS.")
            bot_is_speaking = False
            # Here, you can stop or manage TTS

    async def receive_messages():
        """Receive WebSocket messages and feed audio to STT"""
        nonlocal stream_sid
        try:
            while True:
                message = await websocket.receive_text()
                if bot_is_speaking:
                    await stop_tts()
                    
                message = json.loads(message)
                event_type = message.get("event")
                if event_type == "connected":
                    greeting = "How can I help you today?"
                    audio = await synthesize_text(greeting, lang_code)
                    await speak_pcm(audio, websocket, stream_sid)
                    
                elif event_type == "start":
                    stream_sid = message.get("stream_sid")
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
                    extracted_attributes = extract_dynamic_attributes(txt)
                    # Asynchronously detect the language
                    lang_task = asyncio.create_task(detect_language(txt))
                    # Wait for the language detection result
                    lang_code, confidence = await lang_task

                    # Update context (as in your code)
                    if 'color' in extracted_attributes:
                        user_context['color'] = extracted_attributes['color']
                    if 'fabric' in extracted_attributes:
                        user_context['fabric'] = extracted_attributes['fabric']

                    user_context['lang_code'] = lang_code  # Store language for session

                    logging.info(f"Detected language: {lang_code}")
                    intent, filtered_entities, intent_confidence = await detect_textile_intent_openai(txt, lang_code)
                    logging.info(f"Detected Intent: {intent}")
                    
                    products = []
                    shop_name = "Krishna Textiles"
                    if intent == "product_search":
                        products = await query_products(txt, lang=lang_code)

                    ai_reply = await generate_reply(
                        user_query=txt,
                        products=products,
                        shop_name=shop_name,
                        action=None,
                        language=lang_code,
                        intent=intent,
                    )

                    logging.info(f"AI Reply: {ai_reply}")
                    audio = await synthesize_text(ai_reply, language_code=lang_code)
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
