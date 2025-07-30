import openai
import json
import logging
from typing import Tuple
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()  # Load .env file
openai.api_key = os.getenv("GPT_API_KEY")
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
        response = await openai.ChatCompletion.acreate(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150
        )
        content = response.choices[0].message["content"].strip()
        # Clean JSON response
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        return result["language"], result["confidence"]
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        return "en-IN", 0.5  # Default fallback
# Test function
async def test_language_detection():
    """Test the language detection with sample texts"""
    # Check if API key is set
    if not openai.api_key:
        print(":x: Error: OpenAI API key not found!")
        print("Set it using: export OPENAI_API_KEY='your-key-here'")
        return
    test_cases = [
        "muje lal saree chahiye",           # Hindi Romanized
        "લાલ લહેંગા કેટલાનો છે?",              # Gujarati
        "Do you have silk blouses?",        # English
        "saree ma embroidery che?",         # Mixed Gujarati
        "मुझे लाल साड़ी चाहिए",               # Hindi Devanagari
        "Hello, I want to buy lehenga"      # English
    ]
    print(":thread: Testing Textile Language Detection...")
    print("="*50)
    for i, text in enumerate(test_cases, 1):
        try:
            print(f"\n{i}. Testing: '{text}'")
            language, confidence = await detect_language(text)
            print(f"   → Language: {language}")
            print(f"   → Confidence: {confidence}")
        except Exception as e:
            print(f"   :x: Error: {e}")
    print("\n:white_check_mark: Testing completed!")
# Main function
async def main():
    """Main function to run the language detector"""
    print(":convenience_store: Textile WhatsApp Bot - Language Detector")
    print("="*45)
    # Run tests first
    await test_language_detection()
    # Interactive mode
    print("\n:robot_face: Interactive Mode (type 'quit' to exit):")
    print("-"*40)
    while True:
        try:
            user_input = input("\nEnter message to detect language: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(":wave: Goodbye!")
                break
            if not user_input:
                print(":warning:  Please enter a message!")
                continue
            print(":mag: Detecting language...")
            language, confidence = await detect_language(user_input)
            print(f":memo: Input: '{user_input}'")
            print(f":globe_with_meridians: Detected Language: {language}")
            print(f":bar_chart: Confidence: {confidence:.2f}")
        except KeyboardInterrupt:
            print("\n:wave: Goodbye!")
            break
        except Exception as e:
            print(f":x: Error: {e}")
