import asyncio
import base64
import json
import logging
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# --- Mock STT handler ---
class MockSTTStreamHandler:
    def __init__(self):
        self._queue = asyncio.Queue()
        self._closed = False

    async def send_audio_chunk(self, pcm_chunk: bytes):
        # Just put the chunk into queue (simulate STT receiving audio)
        await self._queue.put(("interim transcript", False, "en-IN"))

    async def get_transcript(self):
        # Simulate waiting for transcript result
        try:
            # Wait for transcript or timeout after 0.2s
            return await asyncio.wait_for(self._queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            # No transcript ready
            return ("", False, "en-IN")

    async def close_stream(self):
        self._closed = True

# --- Mock TTS synthesize ---
async def synthesize_text(text, language_code="en-IN"):
    # Return silent audio bytes of fixed length for testing
    logging.info(f"Synthesizing text: {text} in language: {language_code}")
    return b'\0' * 32000  # silent audio chunk

async def speak_pcm(pcm_audio: bytes, websocket: WebSocket, stream_sid: str):
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

# --- Your AI Analyzer mock ---
class TextileAnalyzer:
    async def analyze_message(self, text, tenant_id=None):
        # Simple echo for demo
        return f"You said: {text}"

analyzer = TextileAnalyzer()

@app.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept(origin=None)
    logging.info("✅ WebSocket connection accepted at /stream")

    stt = MockSTTStreamHandler()
    stream_sid = None
    lang_code = 'en-IN'
    bot_is_speaking = False
    tts_task = None
    tenant_id = None  # no DB here for simplicity

    async def stop_tts():
        nonlocal bot_is_speaking, tts_task
        if bot_is_speaking:
            logging.info(":red_circle: Interrupt detected, stopping TTS.")
            bot_is_speaking = False
            if tts_task:
                tts_task.cancel()
                try:
                    await tts_task
                except asyncio.CancelledError:
                    logging.info("TTS task cancelled")

    async def receive_messages():
        nonlocal stream_sid
        try:
            while True:
                message = await websocket.receive_json()
                if bot_is_speaking:
                    await stop_tts()

                event_type = message.get("event")
                if event_type == "connected":
                    greeting = "How can I help you today?"
                    # No DB lookup here
                    audio = await synthesize_text(greeting, lang_code)
                    await speak_pcm(audio, websocket, stream_sid)

                elif event_type == "start":
                    stream_sid = message.get("stream_sid")
                    logging.info(f"stream_sid: {stream_sid}")

                elif event_type == "media":
                    pcm = base64.b64decode(message["media"]["payload"])
                    await stt.send_audio_chunk(pcm)

                elif event_type == "stop":
                    logging.info("Call ended.")
                    break
        except WebSocketDisconnect:
            logging.info(":octagonal_sign: WebSocket disconnected")

    async def process_transcripts():
        last_activity = time.time()
        while True:
            try:
                txt, is_final, lang = await asyncio.wait_for(stt.get_transcript(), timeout=0.2)
                if is_final and txt:
                    logging.info(f"Final transcript: {txt}")
                    last_activity = time.time()
                    ai_reply = await analyzer.analyze_message(text=txt, tenant_id=tenant_id)
                    logging.info(f"AI Reply: {ai_reply}")

                    nonlocal bot_is_speaking, tts_task
                    bot_is_speaking = True
                    tts_audio = await synthesize_text(ai_reply, language_code=lang_code)
                    tts_task = asyncio.create_task(speak_pcm(tts_audio, websocket, stream_sid))
                    await tts_task
                    bot_is_speaking = False

                elif txt:
                    logging.info(f"Interim transcript: {txt}")
                    last_activity = time.time()

            except asyncio.TimeoutError:
                if time.time() - last_activity > 15:
                    logging.info("No response from user after timeout.")
                    last_activity = time.time()

    try:
        receiver_task = asyncio.create_task(receive_messages())
        processor_task = asyncio.create_task(process_transcripts())
        done, pending = await asyncio.wait([receiver_task, processor_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
    except Exception as e:
        logging.error(f"Error in WebSocket: {e}")
    finally:
        await stt.close_stream()
