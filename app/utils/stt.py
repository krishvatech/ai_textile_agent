# sarvam_stt_stream.py (final patched version with 8kHz -> 16kHz conversion)
import asyncio
import base64
import json
import logging 
import os
import websockets
import audioop
from dotenv import load_dotenv
from websockets.client import connect
load_dotenv()

SARVAM_URL = "wss://api.sarvam.ai/speech-to-text/ws?language-code=unknown"
SARVAM_KEY = os.getenv("SARVAM_API_KEY")

if not SARVAM_KEY:
    raise EnvironmentError("❌ SARVAM_API_KEY not set in environment or .env file!")

class SarvamSTTStreamHandler:
    def __init__(self):
        self.transcript_queue = asyncio.Queue()
        self.ws = None

    async def start_stream(self):
        logging.info(f"🌐 Trying Connecting to Sarvam STT WebSocket...")
        self.ws = await connect(
            SARVAM_URL,
            extra_headers={"api-subscription-key": SARVAM_KEY}
        )
        print("✅ Connected to Sarvam WebSocket")
        logging.info("✅ Connected to Sarvam WebSocket")
        asyncio.create_task(self._receive_transcripts())
        
    async def reset(self):
        self.transcript_queue = asyncio.Queue()

    async def send_audio_chunk(self, pcm: bytes):
        if not self.ws or self.ws.closed:
            logging.warning("⚠️ WebSocket not connected. Trying to reconnect...")
            try:
                await self.start_stream()
            except Exception as reconnect_error:
                logging.error(f"❌ Failed to reconnect WebSocket: {reconnect_error}")
                return
            logging.warning("⚠️ WebSocket not connected. Trying to reconnect...")
            try:
                await self.start_stream()
            except Exception as reconnect_error:
                logging.error(f"❌ Failed to reconnect WebSocket: {reconnect_error}")
                return

        try:
            # Convert 8000 Hz -> 16000 Hz (linear interpolation)
            converted_pcm = audioop.ratecv(pcm, 2, 1, 8000, 16000, None)[0]
            encoded = base64.b64encode(converted_pcm).decode("utf-8")

            message = {
                "audio": {
                    "data": encoded,
                    "sample_rate": 16000,
                    "encoding": "audio/wav"
                }
            }
            await self.ws.send(json.dumps(message))
            # logging.info(f"📤 Sent converted PCM chunk to Sarvam (original: {len(pcm)} bytes → 16kHz)")
        except Exception as e:
            logging.error(f"❌ Failed to send audio to Sarvam: {e}")

    async def _receive_transcripts(self):
        try:
            async for message in self.ws:
                logging.info(f"🌐 Sarvam raw message: {message}")
                try:
                    data = json.loads(message)

                    # ✅ Extract transcript from 'data' type message
                    if data.get("type") == "data" and "transcript" in data.get("data", {}):
                        transcript = data["data"]["transcript"]
                        is_final = bool(transcript.strip())
                        language_code = data["data"].get("language_code", "unknown")  # assume non-empty is final
                        await self.transcript_queue.put((transcript, is_final, language_code))
                        logging.info(f"📝 Transcript queued: {transcript} (is_final={is_final}, language_code={language_code})")

                    elif data.get("type") == "transcript":
                        transcript = data["data"]["transcript"]
                        is_final = data["data"]["is_final"]
                        language_code = data["data"].get("language_code", "unknown")  # assume non-empty is final
                        await self.transcript_queue.put((transcript, is_final, language_code))
                    else:
                        logging.warning(f"📩 Other message (ignored): {data}")
                        
                except Exception as parse_error:
                    logging.error(f"❌ Parse error: {parse_error}")
        except Exception as e:
            logging.error(f"❌ WebSocket receive error: {e}")
            await asyncio.sleep(1)
            await self.start_stream()  # Reconnect
            
            await asyncio.sleep(1)
            await self.start_stream()  # Reconnect
            
    async def get_transcript(self):
        return await self.transcript_queue.get()

    async def close_stream(self):
        if self.ws:
            await self.ws.close()