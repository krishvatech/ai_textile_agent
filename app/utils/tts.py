import os
import aiohttp
import base64
import logging
from dotenv import load_dotenv
import asyncio

load_dotenv() 

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_API_URL = "https://api.sarvam.ai/text-to-speech"


def downsample_to_8k(wav_data: bytes) -> bytes:
    try:
        return wav_data
    except Exception as e:
        logging.error(f"❌ Downsampling failed: {e}")
        return b""
async def synthesize_stream(text: str, language_code: str = "en-IN"):
    audio = await synthesize_text(text, language_code)
    chunk_size = 9600
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk += b'\0' * (chunk_size - len(chunk))
        yield chunk
        
async def synthesize_text(text: str, language_code: str = "en-IN", retries: int = 3, delay: float = 1.0) -> bytes:
    if not text or not isinstance(text, str) or not text.strip():
        logging.warning("⚠️ Empty or invalid text for TTS.")
        return b""
    
    if not SARVAM_API_KEY:
        logging.error("❌ SARVAM_API_KEY not set in environment or .env file!")
        return b""

    payload = {
        "target_language_code": language_code,
        "text": text.strip(),
        "speaker": "anushka",           # ✅ Valid for bulbul:v2
        "model": "bulbul:v2",           # ✅ Must match speaker
        "pitch": 0.0,
        "loudness": 1.0,
        "pace": 0.8,
        "speech_sample_rate": 8000,
        "enable_preprocessing": False
    }

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }

    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=10)
                async with session.post(SARVAM_API_URL, headers=headers, json=payload, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("audios") and isinstance(data["audios"], list):
                            wav_data = base64.b64decode(data["audios"][0])
                            return downsample_to_8k(wav_data)
                        else:
                            logging.warning("🟡 Invalid TTS response format")
                    else:
                        logging.warning(f"🟥 Sarvam TTS HTTP {resp.status}: {await resp.text()}")
        except Exception as e:
            logging.error(f"❌ Sarvam TTS attempt {attempt} failed: {e}")
        await asyncio.sleep(delay)

    logging.error("❌ All Sarvam TTS attempts failed")
    return b""

def get_tts_chunks(pcm_audio: bytes, chunk_size: int = 9600) -> list:
    chunks = []
    total_len = len(pcm_audio)
    for i in range(0, total_len, chunk_size):
        chunk = pcm_audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk += b'\0' * (chunk_size - len(chunk))
        chunks.append(base64.b64encode(chunk).decode("ascii"))
    return chunks
