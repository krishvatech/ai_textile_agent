import asyncio
from openai import AsyncOpenAI
import json
import logging
from typing import Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GPT_API_KEY")
client = AsyncOpenAI(api_key=api_key)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to detect language using GPT-5 Mini (no temperature/max_tokens params)
async def detect_language(text: str, last_language: str) -> Tuple[str, float]:
    """
    Detect language using OpenAI API for textile bot
    Supports: Hindi, Gujarati, English (including Romanized)
    """
    prompt = f"""You are a language detection expert for an Indian textile business WhatsApp bot.
**Task**: Detect the language of the customer's message below. Supported languages are:
- Hindi (including Devanagari script and Romanized Hindi)
- Gujarati (including Gujarati script and Romanized Gujarati)
- English (Indian English)
**Special Instructions**:
1. If the message is a simple greeting or chit-chat like "hello", "hi", "how are you", do NOT finalize the language based on this message. Instead, treat this as neutral and wait for a meaningful message.
2. For the first meaningful message after greetings (such as textile-related requests), detect the language properly and consider this the conversation's fixed language.
3. After the conversation language is fixed, assume all subsequent messages are in that language unless explicitly switched.
4. Handle Romanized inputs accurately. For example, "muje sadi chahiye" is Hindi (Romanized), "mare sadi joi che" is Gujarati (Romanized).
5. For mixed or code-switched messages, choose the dominant language.
**Examples**:
- "hello" → return language as `"neutral"` or indicate language not finalized
- "muje lal saree chahiye" → Hindi (Romanized)
- "mare sadi joi che" → Gujarati (Romanized)
- "Do you have silk blouses?" → English
- "saree ma embroidery che?" → Gujarati (Mixed)
**Response Format** (JSON only):
{{
  "language": "<hi-IN|gu-IN|en-IN|neutral>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}}
**Customer Message**: "{text}"
"""
    try:
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            # Important: don't send temperature/max_tokens to gpt-5-mini
        )
        content = resp.choices[0].message.content.strip()
        # Clean possible code fences
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        language = result.get("language", "en-IN")
        confidence = float(result.get("confidence", 0.5))
        return language, confidence
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        # Fall back to the last known language if available
        return (last_language or "en-IN"), 0.5