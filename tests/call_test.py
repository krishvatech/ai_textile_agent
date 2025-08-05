# local_text_processor.py
# This is an updated standalone Python script that takes text input from the user,
# detects language, extracts attributes, detects intent, queries products (if applicable),
# and generates an AI reply.
# It imports the actual functions from your app modules instead of using placeholders.
# Assumption: This script is placed in a location where it can import from the 'app' package.
# If your project structure requires adjustments (e.g., sys.path modifications), add them accordingly.
# Note: This doesn't include TTS or STT as it's text-based; it focuses on intent, language, and AI reply.

import asyncio
import logging
import json
import sys
import os

# Adjust sys.path if needed to import from app (e.g., if this script is outside the app directory)
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Uncomment and adjust if necessary

from app.core.lang_utils import detect_language
from app.core.attribute_extraction import extract_dynamic_attributes
from app.core.intent_utils import detect_textile_intent_openai
from app.utils.pinecone_utils import query_products
from app.core.ai_reply import TextileAnalyzer

analyzer = TextileAnalyzer()

# User context (as in the original code)
user_context = {}

async def process_text_input(txt):
    """Process the text input similar to the WebSocket processor."""
    logging.info(f"Processing input: {txt}")
    
    # Extract attributes
    extracted_attributes = extract_dynamic_attributes(txt)
    
    # Detect language
    lang_code, confidence = await detect_language(txt)
    
    # Update context
    if 'color' in extracted_attributes:
        user_context['color'] = extracted_attributes['color']
    if 'fabric' in extracted_attributes:
        user_context['fabric'] = extracted_attributes['fabric']
    
    # Detect intent
    intent, filtered_entities, intent_confidence = await detect_textile_intent_openai(txt, lang_code)
    logging.info(f"Detected language: {lang_code}")
    logging.info(f"Detected Intent: {intent}")
    
    # Query products if needed
    products = []
    shop_name = "Krishna Textiles"
    if intent == "product_search":
        products = await query_products(txt, lang=lang_code)
    
    # Generate AI reply
    ai_reply = await analyzer.analyze_message(
            user_query=txt,
            tenant_id=12,
    )
    
    logging.info(f"AI Reply: {ai_reply}")
    return ai_reply

# Main function to run the script
async def main():
    logging.basicConfig(level=logging.INFO)
    print("Enter your text input (type 'exit' to quit):")
    while True:
        txt = input("> ")
        if txt.lower() == 'exit':
            break
        reply = await process_text_input(txt)
        print(f"Reply: {reply}")

if __name__ == "__main__":
    asyncio.run(main())
