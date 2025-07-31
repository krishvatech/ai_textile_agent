from openai import AsyncOpenAI
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GPT_API_KEY")
client = AsyncOpenAI(api_key=api_key)

async def generate_reply(
    user_query: str,
    products: List[Dict],
    shop_name: str,
    action: Optional[str] = None,
    language: str = "en",
    intent: Optional[str] = None,
) -> str:
    prompt = f"""
    You are a polite, expert textile shop assistant from India.
    The user's detected intent is: {intent}.
    Always reply in the {language} language.
    Keep replies short—no longer than 2-3 WhatsApp message lines, max 2 sentences.
    Start with the main point and only add ONE clarifying detail or follow-up question.
    If the intent is 'product_search', confirm if the product is available and ask for fabric/design preference.
    If the intent is 'greeting', just give a friendly greeting and ask how you can help.
    If the intent is 'goodbye', wish the customer well in a friendly, short way.
    If you need more info, request it briefly and politely in the same language.
    User: "{user_query}"
    Assistant:
    """
    if not products:
        prompt += f"No matching products found.\n"
    else:
        prods = "\n".join(
            f"{i+1}. {p['name']} – ₹{p['price']} (Color: {p['color']})"
            for i, p in enumerate(products)
        )
        prompt += f"Matching products:\n{prods}\n"

    if action == "order":
        prompt += "User wants to place an order. Ask politely for the delivery address.\n"
    elif action == "rental":
        prompt += "User wants to rent. Ask for rental dates.\n"

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a warm, polite, and chatty textile shop owner who loves helping and chatting with customers, "
                        "whether it's about shopping or just casually talking."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Sorry, I couldn't process your message due to an error: {e}"
