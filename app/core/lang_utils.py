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

# Function to detect language using OpenAI API
async def detect_language(text: str,last_language: str) -> Tuple[str, float]:
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
5. If the message contains common textile-related words such as "xl", "xxl", "wedding", "cotton" or similar,
   and the rest of the message is mostly in a local language (Hindi or Gujarati),
   do NOT classify the message as English based only on those words.
   However, if the message is clearly written in English grammar and vocabulary (e.g. "I need for cotton"),
   classify it as English ("en-IN").
5.a. If the message contains any color names (e.g., "red", "blue", "green", "yellow", "pink", etc.),
     treat these cases as ambiguous and return the language as the last detected language: "{last_language}".
6. If the message is written in Gujarati or Hindi script but contains transliterations or loanwords of common English textile-related terms 
   (e.g., Gujarati word "મેરેજ" which is transliteration of "marriage"),
   treat these cases as ambiguous and return the language as the last detected language: "{last_language}"
7. If the message contains mostly English words but written in Hindi or Gujarati script (transliterated English),
   you may classify as English ("en-IN") to better reflect spoken language.
**Examples**:
- "muje lal saree chahiye" → Hindi (Romanized)
- "લાલ લહેંગા કેટલાનો છે?" → Gujarati
- "Do you have silk blouses?" → English
- "saree ma embroidery che?" → Gujarati (Mixed)
- "મેરેજ" → Use last detected language: "{last_language}"
- "xl size chahiye" → Use last detected language: "{last_language}"
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
            model="gpt-4.1-mini",
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
        return "en-IN", 0.5
    
async def main():
    last_language = "en-IN"  # default initial last language
    print("Textile Bot Language Detector (type 'exit' to quit)")
    while True:
        transcript = input("\nEnter transcript: ").strip()
        if transcript.lower() == "exit":
            print("Exiting...")
            break
        lang, conf = await detect_language(transcript, last_language)
        print(f"Detected Language: {lang}, Confidence: {conf:.2f}")
        last_language = lang  # update last detected language for next input

if __name__ == "__main__":
    asyncio.run(main())