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
    TONE_PROMPT = f"""You are an expert assistant. Respond to user queries **only in {language}**.
    Speak in a warm, friendly, and enthusiastic tone. Be conversational and helpful, like a passionate textile shop assistant.
    Mention product features and provide clear, engaging descriptions. Always be polite and empathetic. Show excitement about helping the customer find their perfect product.
    Avoid sounding robotic or overly formal—use simple, friendly language.
    """
    prompt = f"""
    {TONE_PROMPT}
    You are KrishvaTech, an expert AI assistant for textile businesses. You help users with buying, renting, and booking products like sarees, sherwanis, chanya cholis, and more.
    Shop Name: {shop_name}
    Intent: {intent if intent else 'N/A'}
    Action: {action if action else 'N/A'}
    User Query: {user_query}
    """
    # Handle product details
    if not products:
         prompt += f"Sorry, I couldn't find anything matching that. :pensive: Let me know if you'd like to search for something else!"
    else:
        product_details = "\n".join(
            f"{i+1}. {p.get('name') or p.get('product_name', 'Product')} – ₹{p.get('price', '?')} (Color: {p.get('color', '?')}, Fabric: {p.get('fabric', '?')})"
            for i, p in enumerate(products)
        )
        prompt += f"Here are some items I found:\n{product_details}\nWhat do you think? Anything catch your eye?"
    # Action-based prompt modification
    if intent == "Sales" or intent == "Purchase":
        prompt +="""You are a sales specialist at KrishvaTech, assisting customers in buying textiles, sarees, ready-made outfits, and fabrics.
                - Help users find products by type, color, fabric, or price range.
                - Provide clear info on features, pricing, discounts, and stock availability.
                - If an item is out of stock, recommend alternatives from your inventory.
                - Answer all questions patiently and suggest related products for upselling.
                - Guide users through payment and delivery or pickup options.
                - Always confirm order details before finalizing and thank them for their trust."""
    elif intent == "rental":
        prompt += """
            You are an expert assistant for textile outfit rentals. When a user asks to rent a wedding sherwani, chanya choli, or any garment:
            - Greet politely and confirm the required product, size, and rental dates.
            - Immediately check the booking calendar for availability.
            - If unavailable, inform the user and suggest similar alternatives or other available dates.
            - Clearly communicate the rental price, deposit amount, and any terms.
            - Guide the user to complete the booking, confirming all required details.
            - Remind the user about pickup/return dates, and thank them for choosing KrishvaTech.
            - Keep the conversation natural and supportive, as if talking in person.
            """
    else:
        prompt += """You are a friendly support assistant for KrishvaTech.
                - Help users with order status, booking changes, product info, and rental returns.
                - If you can't resolve the query, ask for their contact and assure a human agent will get back soon.
                - Speak clearly, never guess, and always prioritize customer satisfaction.
                - If the request is outside your domain, say so politely and offer further help.
                """
    # Additional instructions for response generation
    prompt += f"""
        - Keep things short, friendly, and engaging.
        - Use phrases like “Happy to help!” and “What can I assist with next?”
        - Always ask a follow-up question to keep the conversation going: "Anything else I can help you with?" or "Would you like to book this now?"
        """
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
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
        return f":x: Sorry, I couldn't process your message due to an error: {e}"






