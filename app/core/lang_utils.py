

# import re
# import openai
# import os
# import json
# import logging
# from typing import Tuple
# from dotenv import load_dotenv
# from openai import AsyncOpenAI

# load_dotenv()
# api_key = os.getenv("GPT_API_KEY")
# client = AsyncOpenAI(api_key=api_key)

# def fast_script_detect(text: str):
#     if re.search(r"[\u0A80-\u0AFF]", text):    # Gujarati script
#         return "gu-IN", 1.0
#     if re.search(r"[\u0900-\u097F]", text):    # Devanagari script (Hindi)
#         return "hi-IN", 1.0
#     # English if all words are ascii (maybe add extra filtering for very short text)
#     if re.fullmatch(r"[A-Za-z0-9 ,.'\"?!\-@]+", text) and len(text.split()) > 1:
#         # Could still be Romanized, so only label as English if common English words detected
#         # Optionally: check for presence of common Hindi/Gujarati words in Roman
#         return None  # Ambiguous; send to GPT
#     return None  # Ambiguous or mixed

# async def detect_language(text: str) -> Tuple[str, float]:
#     # Step 1: Fast detection (script-based)
#     result = fast_script_detect(text)
#     if result:
#         return result

#     # Step 2: Fallback to GPT (your current method)
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
#         response = await client.chat.completions.create(
#             model="gpt-4-1-mini",  # Use your actual model name
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.2,
#             max_tokens=150
#         )

#         content = response.choices[0].message.content.strip()
#         if content.startswith("```"):
#             content = content.replace("```json", "").replace("```", "").strip()

#         result = json.loads(content)
#         language = result["language"].lower()
#         confidence = result["confidence"]

#         supported_languages = {
#             "hi": "hi-IN",
#             "gu": "gu-IN",
#             "en": "en-IN",
#         }
#         if language not in supported_languages:
#             logging.warning(f"Detected language '{language}' is unsupported, falling back to 'en-IN'.")
#             language = "en"

#         return supported_languages[language], confidence
#     except Exception as e:
#         logging.error(f"Language detection failed: {e}")
#         return "en-IN", 0.5


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

def is_gujarati_script(text: str):
    return bool(re.search(r"[\u0A80-\u0AFF]", text))

def is_devanagari_script(text: str):
    return bool(re.search(r"[\u0900-\u097F]", text))

async def detect_language(text: str) -> Tuple[str, float]:
    # 1. Fast Devanagari check (Hindi)
    if is_devanagari_script(text):
        return "hi-IN", 1.0

    # 2. For Gujarati script, let GPT decide English vs. Gujarati
    if is_gujarati_script(text):
        prompt = f"""
You are a language detector for an Indian textile WhatsApp bot.
If the message is written in Gujarati script, check if it's just English words spelled in Gujarati script (e.g., 'આઈ વોન્ટ રેડ તારી').
- If so, classify as English ("en").
- If it contains real Gujarati words or grammar, classify as Gujarati ("gu").
If the message is in Roman script, classify as "en", "hi", or "gu" as appropriate.
Respond as JSON only:
{{
  "language": "<gu|en>",
  "confidence": <0.0-1.0>
}}
Message: "{text}"
"""
        try:
            response = await client.chat.completions.create(
                model="gpt-4-1106-preview",  # Use a valid model name
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=60,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            language = result["language"].lower()
            confidence = result["confidence"]
            lang_map = {"gu": "gu-IN", "en": "en-IN"}
            return lang_map.get(language, "en-IN"), confidence
        except Exception as e:
            logging.error(f"Language detection failed: {e}")
            return "en-IN", 0.5

    # 3. For Roman/Latin script, use your normal GPT prompt
    prompt = f"""
You are a language detector for an Indian textile WhatsApp bot.
Supported: Hindi, Gujarati, English (native or Romanized).
Classify: "en" for English, "hi" for Hindi, "gu" for Gujarati.
Reply as JSON only:
{{
  "language": "<en|hi|gu>",
  "confidence": <0.0-1.0>
}}
Message: "{text}"
"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4-1106-preview",  # Use a valid model name
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=60,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        language = result["language"].lower()
        confidence = result["confidence"]
        lang_map = {"gu": "gu-IN", "hi": "hi-IN", "en": "en-IN"}
        return lang_map.get(language, "en-IN"), confidence
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        return "en-IN", 0.5
