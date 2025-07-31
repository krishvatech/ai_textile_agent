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

    Act like a friendly, casual Indian shop owner who loves chatting with customers. 
    Handle the conversation in a human, down-to-earth way — short, warm, and natural.

    📦 If the user is asking about products, rentals, or buying — show matching products if available and offer help.

    👋 If the user is asking casually (e.g., "What do you do?", "How are you?", "What's up?") — reply in a fun, light way, like a local shopkeeper would. Don't give a long introduction about the store unless needed.

    😊 End with a friendly follow-up question or comment to keep the chat going.

    💬 Respond only in {language}. Keep tone casual, not formal."""
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
