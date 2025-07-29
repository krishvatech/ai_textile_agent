import os
import httpx

EXOTEL_SID = os.getenv("EXOTEL_SID")
EXOTEL_TOKEN = os.getenv("EXOTEL_TOKEN")

async def send_voice_response(call_sid, text, lang="en"):
    # You would use Exotel's TTS or your own TTS to generate and send audio
    # Here is a placeholder
    print(f"Send TTS for call {call_sid}: {text} [{lang}]")
