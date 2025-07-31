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
    language: str = "en"
) -> str:
    prompt = f"""Shop name: {shop_name}
    User query: {user_query}
    Language: {language}

    You're the owner of a textile shop called {shop_name}. You're helpful, polite, and always guide customers about sarees, fabric, rentals, or placing an order.

    📦 If the user asks about clothes, orders, or rentals — reply helpfully and show available products (if provided). Ask a follow-up question to guide them.

    🛍️ If the user's message is unrelated to shopping (like talking about food, greetings, or general conversation), gently steer the chat back to textile shopping. Do **not** talk about your snacks, tea breaks, or anything unrelated. Instead, respond warmly and ask:
    > “Are you looking for any particular saree or fabric today?” or similar.

    😊 Always end with a friendly, shopping-related follow-up to keep the user engaged.

    💬 Reply only in {language}. Keep tone polite and clear."""
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
