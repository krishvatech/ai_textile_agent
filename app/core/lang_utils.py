import openai
import os
import json
import logging
from typing import Tuple
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()  # Load .env file
api_key = os.getenv("GPT_API_KEY")
client = AsyncOpenAI(api_key=api_key)

async def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the language using OpenAI API for textile bot.
    Supports: Hindi, Gujarati, English (including Romanized).
    """
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
        # Send request to OpenAI to detect language
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",  # Use your actual model name
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150
        )

        content = response.choices[0].message.content.strip()
        # Clean JSON response
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        result = json.loads(content)
        language = result["language"]
        confidence = result["confidence"]

        # Dynamically handle the language detection without hardcoding fallback
        supported_languages = {
            "hi": "hi-IN",  # Hindi
            "gu": "gu-IN",  # Gujarati
            "en": "en-IN",  # English
        }

        # If the detected language is in the supported list, return it; otherwise, default to English.
        language = result["language"].lower()  # just in case model uses uppercase
        if language not in supported_languages:
            logging.warning(f"Detected language '{language}' is unsupported, falling back to 'en-IN'.")
            language = "en"

        return supported_languages[language], confidence
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        return "en-IN", 0.5  # Default fallback to English if the detection fails

# import openai
# import os
# import json
# import logging
# from typing import Tuple
# from dotenv import load_dotenv
# from openai import AsyncOpenAI
# import datetime
# import re
# from typing import Any, Tuple

# # Load environment variables
# load_dotenv()  # Load .env file
# api_key = os.getenv("GPT_API_KEY")
# client = AsyncOpenAI(api_key=api_key)

# # ---------------------- Unified GPT Transcript Analyzer ----------------------
# async def analyze_transcript_gpt(transcript: str) -> Tuple[str, str]:
#     """
#     Analyze the transcript, clean it, and detect the language code (e.g., en-IN, hi-IN, gu-IN).
#     """
#     logging.info("in analyze_transcript_gpt")

#     start_time = datetime.utcnow()
    
#     prompt = f"""
#     You are a multilingual voicebot assistant.
#     Given the user's raw voice transcript, perform the following tasks:
#     1. Detect the language of the user's message (English, Hindi, or Gujarati).
#     2. Return the cleaned transcript and the language code in the format 'en-IN', 'hi-IN', or 'gu-IN'.
#     3. Do not generate any AI responses. Only return the language code and cleaned transcript.

#     User Transcript: "{transcript}"
#     """
    
#     try:
#         # Using the new OpenAI ChatCompletion API method (for version 1.95.1)
#         response = await client.chat.completions.create(
#             model="gpt-4.1-mini",  # Use your actual model name
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.2,
#             max_tokens=150
#         )

#         content = response.choices[0].message.content.strip()

#         # Clean the response if needed
#         if content.startswith("```"):
#             content = content.replace("```json", "").replace("```", "").strip()

#         # Parse the cleaned response
#         data = json.loads(content)
#         logging.info(f"Processed transcript: {data['transcript']}, Language code: {data['lang_code']}")

#         end_time = datetime.utcnow()
#         logging.info(f":white_check_mark: analyze_transcript_gpt END at {end_time.isoformat()}")
#         logging.info(f":clock3: Total duration: {(end_time - start_time).total_seconds():.2f} seconds")
        
#         # Return cleaned transcript and dynamically validated language code
#         lang_code = await validate_language_code(data['lang_code'])
#         return data["transcript"], lang_code
    
#     except Exception as e:
#         logging.error(f":x: analyze_transcript_gpt failed: {e}")
#         return transcript, "en-IN"  # Default to English if failed


# # ---------------------- Language Normalization ----------------------
# async def detect_language(transcript: str) -> Tuple[str, str]:
#     """
#     Normalize the spoken transcript and detect the language.
#     Returns the cleaned sentence and detected language code.
#     """
#     logging.info("in normalize_and_detect_language")

#     prompt = f"""
#     You are an expert assistant for Indian voice AI.
#     :star2: Task:
#     1. Normalize the spoken sentence from noisy STT output.
#     2. Identify if the spoken language is Romanized (using Latin script).
#     3. Identify the true spoken language (English, Hindi, or Gujarati) and return the language code.
#     4. Return the cleaned sentence in the correct language's native script.
#     5. Return language code: 'en-IN', 'hi-IN', or 'gu-IN'.

#     :blue_book: Format your reply in this exact JSON:
#     {{
#         "transcript": "<cleaned sentence>",
#         "lang_code": "<en-IN|hi-IN|gu-IN>"
#     }}

#     :date: Input: "{transcript}"
#     Only return valid JSON. Do not explain.
#     """
    
#     try:
#         # Using the new OpenAI ChatCompletion API method (for version 1.95.1)
#         response = await client.chat.completions.create(
#             model="gpt-4.1-mini",  # Use your actual model name
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.2,
#             max_tokens=150
#         )

#         content = response.choices[0].message.content.strip()
#         if content.startswith("```"):
#             match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
#             if match:
#                 content = match.group(1)

#         result = json.loads(content)

#         # Handle dynamic language code validation
#         lang_code = await validate_language_code(result["lang_code"])
#         return result["transcript"], lang_code
    
#     except Exception as e:
#         logging.error(f":x: normalize_and_detect_language failed: {e}")
#         return transcript, "en-IN"  # Fallback to English if any error occurs


# # ---------------------- Language Code Validation ----------------------
# async def validate_language_code(detected_lang: str) -> str:
#     """
#     Dynamically map detected language to supported language codes (only for Hindi, Gujarati, English).
#     Handles Romanized text, mixed languages, and fallback to supported languages.
#     """
#     # Supported languages for Sarvam TTS
#     supported_languages = {
#         "hi": "hi-IN",  # Hindi
#         "gu": "gu-IN",  # Gujarati
#         "en": "en-IN",  # English
#     }

#     # Check if the detected language is supported
#     if detected_lang in supported_languages:
#         return supported_languages[detected_lang]

#     # If the detected language is not supported, fallback to `en-IN` (English)
#     logging.warning(f"Detected language '{detected_lang}' is unsupported, falling back to 'en-IN'.")
#     return "en-IN"  # Default to English if unsupported language detected
