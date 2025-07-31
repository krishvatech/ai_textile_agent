from openai import AsyncOpenAI
import json
import logging
from typing import Tuple
import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file
api_key=os.getenv("GPT_API_KEY")
client = AsyncOpenAI(api_key=api_key)
async def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect language using OpenAI API for textile bot
    Supports: Hindi, Gujarati, English (including Romanized)
    """
    prompt = f"""
You are a language detection expert for an Indian textile business WhatsApp bot.
**Task**: Detect the language of the customer message below.
**Supported Languages**:
- Hindi (Devanagari script or Romanized)
- Gujarati (Gujarati script or Romanized)
- English (Indian English)
**Special Instructions**:
1. Handle Romanized Hindi/Gujarati (written in English letters)
2. Consider textile business context and terminology
3. For mixed language, choose the dominant language
4. Account for common Indian English + local language mixing
**Examples**:
- "muje lal saree chahiye" → Hindi (Romanized)
- "લાલ લહેંગા કેટલાનો છે?" → Gujarati
- "Do you have silk blouses?" → English
- "saree ma embroidery che?" → Gujarati (Mixed)
**Response Format** (JSON only):
{{
    "language": "<hi-IN|gu-IN|en-IN>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>"
}}
**Customer Message**: "{text}"
"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4-1106-preview",  # or your actual model name
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        # Clean JSON response
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        return result["language"], result["confidence"]
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        return "en-IN", 0.5  # Default fallback

