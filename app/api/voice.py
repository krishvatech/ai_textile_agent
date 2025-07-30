from fastapi import APIRouter, Request, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from app.db.session import get_db
from app.core.ai_agent import handle_user_message
from app.utils.exotel_utils import send_voice_response
from app.utils.stt import SarvamSTTStreamHandler
from app.utils.tts import synthesize_stream
import json
import asyncio
import logging
import base64
from fastapi.responses import Response
import asyncio


router = APIRouter()

# @router.post("/webhook", operation_id="voice_webhook_post_1")
# async def voice_webhook(req: Request, background_tasks: BackgroundTasks, db = Depends(get_db)):
#     print("Received voice call webhook!")

#     body = await req.json()
#     user_id = body.get("From")
#     transcript = body.get("SpeechResult", "")
#     tenant_id = 1  # Lookup based on incoming number/DB mapping
#     shop_name = "your shop name"
#     call_sid = body.get("CallSid")
#     reply_text, _, _ = await handle_user_message(user_id, transcript, tenant_id, shop_name, db, channel="voice")
#     background_tasks.add_task(send_voice_response, call_sid, reply_text, "en")
#     return {"status": "voice_replied"}



@router.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()

    greeting_message = "Hello, Good Morning!"
    logging.info(f"Sending greeting message: {greeting_message}")

    # Send greeting audio chunks via TTS
    async for audio_chunk in synthesize_stream(greeting_message, language_code="en-IN"):
        await websocket.send_bytes(audio_chunk)
        logging.info("Sent greeting audio chunk")

    # Wait to ensure client receives and plays greeting
    await asyncio.sleep(3.0)

    stt_handler = SarvamSTTStreamHandler()
    await stt_handler.start_stream()

    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                logging.warning("Client disconnected gracefully")
                break

            # Handle received messages
            if "bytes" in message:
                pcm = message["bytes"]
                await stt_handler.send_audio_chunk(pcm)

            elif "text" in message:
                try:
                    json_data = json.loads(message["text"])
                    event = json_data.get("event")

                    if event == "media":
                        pcm = base64.b64decode(json_data["media"]["payload"])
                        await stt_handler.send_audio_chunk(pcm)

                    elif event == "connected":
                        logging.info("🔗 Client connected")

                    elif event == "start":
                        logging.info("🚀 Start event received")

                except Exception as e:
                    logging.warning(f"⚠️ Failed to parse text message: {e}")

            # Try to pull final transcript
            try:
                transcript, is_final, lang = await asyncio.wait_for(
                    stt_handler.get_transcript(), timeout=1.5
                )
                if is_final and transcript.strip():
                    logging.info(f"✅ Final Transcript: {transcript}")

                    # Replace user_id, tenant_id, shop_name, db as per your logic
                    reply_text, _, _ = await handle_user_message(
                        user_id="stream_user",
                        message=transcript,
                        tenant_id=1,
                        shop_name="your shop name",
                        db=None,
                        channel="voice"
                    )

                    # Send AI reply audio back to client
                    async for audio_chunk in synthesize_stream(reply_text, language_code=lang):
                        await websocket.send_bytes(audio_chunk)

                    await websocket.close()
                    break

            except asyncio.TimeoutError:
                continue

    except Exception as e:
        logging.error(f"❌ Error in stream handler: {e}")

    finally:
        await stt_handler.close_stream()
        logging.info("Closed STT stream and WebSocket handler")
