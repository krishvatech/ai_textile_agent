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
    You are KrishvaTech, an expert AI assistant for textile businesses. You help users with buying, renting, and booking products like sarees, sherwanis, chanya cholis, and more.
    Shop Name: {shop_name}
    Intent: {intent if intent else 'N/A'}
    Action: {action if action else 'N/A'}
    User Query: {user_query}
    """
    # Handle product details
    if not products:
        prompt += f"No matching products found.\n"
    else:
        product_details = "\n".join(
            f"{i+1}. {p.get('name') or p.get('product_name', 'Product')} – ₹{p.get('price', '?')} (Color: {p.get('color', '?')}, Fabric: {p.get('fabric', '?')})"
            for i, p in enumerate(products)
        )
        prompt += f"Matching products:\n{product_details}\n"
    # Action-based prompt modification
    if action == "order":
        prompt += "User wants to place an order. Ask politely for the delivery address.\n"
    elif action == "rental":
        prompt += """
            You are an expert assistant for textile outfit rentals. When a user asks to rent a wedding sherwani, chanya choli, or any garment:
            - Greet politely and confirm the required product, size, and rental dates.
            - Immediately check the booking calendar for availability.
            - If unavailable, inform the user and suggest similar alternatives or other available dates.
            - Clearly communicate the rental price, deposit amount, and any terms.
            - Guide the user to complete the booking, confirming all required details.
            - Remind the user about pickup/return dates, and thank them for choosing KrishvaTech.
            - Keep the conversation natural and supportive, as if talking in person.\n
            """
    # Additional instructions for response generation
    prompt += f"""
    - Respond in a natural, friendly, human-like tone.
    - Guide users through the full process: product discovery, price inquiries, booking, rental details, payment, delivery, and support.
    - If a user wants to rent an item, always check the availability for their desired date(s), inform them if it's already booked, and suggest alternatives if needed.
    - Always confirm all important details (dates, sizes, colors, rental price, deposit, pickup/return info) before finalizing bookings.
    - If a user wants to buy, give clear info about product features, stock, pricing, and next steps.
    - Proactively clarify doubts, offer recommendations, and upsell relevant products/services.
    - NEVER invent products, prices, or policies. Always use data from the inventory/database.
    - Speak in the user's language: {language}.
    - If the user asks for human support, offer to connect them or record their query for a callback.
    - For every interaction, act as a smart, polite, and knowledgeable textile store representative, ensuring customer delight and trust.
    If you don't know the answer, say, "I will check with our team and get back to you."
    Always end conversations with a call-to-action, like:
    "Would you like to book this now?" or "Is there anything else I can help you with?"
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
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
        # return response.choices[0].message["content"].strip()
        ai_reply = response['choices'][0]['message']['content'].strip()
        return ai_reply
    except Exception as e:
        return f":x: Sorry, I couldn't process your message due to an error: {e}"






