# import openai
# import os
# import json
# import logging
# from typing import Tuple
# from dotenv import load_dotenv
# from openai import AsyncOpenAI

# # Load environment variables
# load_dotenv()  # Load .env file
# api_key = os.getenv("GPT_API_KEY")
# client = AsyncOpenAI(api_key=api_key)

# async def detect_language(text: str) -> Tuple[str, float]:
#     """
#     Detect the language using OpenAI API for textile bot.
#     Supports: Hindi, Gujarati, English (including Romanized).
#     """
#     prompt = f"""
# You are an expert language detection model for an Indian textile business WhatsApp bot.
# **Task**: Detect the primary language of the customer message below.
# **Supported Languages**:
# - Hindi (Devanagari or Romanized)
# - Gujarati (Gujarati or Romanized)
# - English (Indian English, including any Romanized text or fully English sentences)

# **Important**:
# - If the sentence is fully or mostly English, label it "en".
# - If the sentence mixes English with Hindi or Gujarati, label the dominant Indian language ("hi" or "gu").
# - If the input is Romanized, decide based on context which Indian language it represents.
# - Do NOT label as Hindi or Gujarati if the text is clearly English.

# **Examples**:
# - "I want a red saree" → English
# - "muje lal saree chahiye" → Hindi (Romanized)
# - "લાલ લહેંગા કેટલાનો છે?" → Gujarati
# - "Do you have silk blouses?" → English
# - "saree ma embroidery che?" → Gujarati (Mixed)

# **Response Format** (JSON only):
# {{
#     "language": "<hi|gu|en>",
#     "confidence": <0.0-1.0>,
#     "reasoning": "<brief explanation>"
# }}

# Customer Message: "{text}"
# """

#     try:
#         # Send request to OpenAI to detect language
#         response = await client.chat.completions.create(
#             model="gpt-4.1-mini",  # Use your actual model name
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.2,
#             max_tokens=150
#         )

#         content = response.choices[0].message.content.strip()
#         # Clean JSON response
#         if content.startswith("```"):
#             content = content.replace("```json", "").replace("```", "").strip()

#         result = json.loads(content)
#         language = result["language"]
#         confidence = result["confidence"]

#         # Dynamically handle the language detection without hardcoding fallback
#         supported_languages = {
#             "hi": "hi-IN",  # Hindi
#             "gu": "gu-IN",  # Gujarati
#             "en": "en-IN",  # English
#         }

#         # If the detected language is in the supported list, return it; otherwise, default to English.
#         language = result["language"].lower()  # just in case model uses uppercase
#         if language not in supported_languages:
#             logging.warning(f"Detected language '{language}' is unsupported, falling back to 'en-IN'.")
#             language = "en"

#         return supported_languages[language], confidence
#     except Exception as e:
#         logging.error(f"Language detection failed: {e}")
#         return "en-IN", 0.5  # Default fallback to English if the detection fails


import re
import openai
import os
import json
import logging
from typing import Tuple
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
api_key = os.getenv("GPT_API_KEY")
client = AsyncOpenAI(api_key=api_key)

def fast_script_detect(text: str):
    if re.search(r"[\u0A80-\u0AFF]", text):    # Gujarati script
        return "gu-IN", 1.0
    if re.search(r"[\u0900-\u097F]", text):    # Devanagari script (Hindi)
        return "hi-IN", 1.0
    # English if all words are ascii (maybe add extra filtering for very short text)
    if re.fullmatch(r"[A-Za-z0-9 ,.'\"?!\-@]+", text) and len(text.split()) > 1:
        # Could still be Romanized, so only label as English if common English words detected
        # Optionally: check for presence of common Hindi/Gujarati words in Roman
        return None  # Ambiguous; send to GPT
    return None  # Ambiguous or mixed

async def detect_language(text: str) -> Tuple[str, float]:
    # Step 1: Fast detection (script-based)
    result = fast_script_detect(text)
    if result:
        return result

    # Step 2: Fallback to GPT (your current method)
    prompt = f"""
You are an expert language detection model for an Indian textile business WhatsApp bot.
**Task**: Detect the primary language of the customer message below.
**Supported Languages**:
- Hindi (Devanagari or Romanized)
- Gujarati (Gujarati or Romanized)
- English (Indian English, including any Romanized text or fully English sentences)

**Important**:
- If the sentence is fully or mostly English, label it "en".
- If the sentence mixes English with Hindi or Gujarati, label the dominant Indian language ("hi" or "gu").
- If the input is Romanized, decide based on context which Indian language it represents.
- Do NOT label as Hindi or Gujarati if the text is clearly English.

**Examples**:
- "I want a red saree" → English
- "muje lal saree chahiye" → Hindi (Romanized)
- "લાલ લહેંગા કેટલાનો છે?" → Gujarati
- "Do you have silk blouses?" → English
- "saree ma embroidery che?" → Gujarati (Mixed)

**Response Format** (JSON only):
{{
    "language": "<hi|gu|en>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>"
}}

Customer Message: "{text}"
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4-1-mini",  # Use your actual model name
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        result = json.loads(content)
        language = result["language"].lower()
        confidence = result["confidence"]

        supported_languages = {
            "hi": "hi-IN",
            "gu": "gu-IN",
            "en": "en-IN",
        }
        if language not in supported_languages:
            logging.warning(f"Detected language '{language}' is unsupported, falling back to 'en-IN'.")
            language = "en"

        return supported_languages[language], confidence
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        return "en-IN", 0.5

