from openai import AsyncOpenAI
import os
import json
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
    language: str = "en"
) -> str:
    prompt = f"""Shop name: {shop_name}
User query: {user_query}
Language: {language}

Check if the query is related to shopping, clothes, sarees, rental, or ordering:
- If YES: Show matching products (if any) and reply as a warm, helpful, polite shop owner. If action is 'order', ask for delivery address. If 'rental', ask for rental dates.
- If NO: still reply in a warm, chatty tone like a friendly shop owner who enjoys talking with customers. Don't end the conversation — ask something casual or say something nice.
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
