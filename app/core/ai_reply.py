import openai
import os

openai.api_key = os.getenv("GPT_API_KEY")

def generate_reply(user_query, products, shop_name, language="en"):
    if not products:
        return f"Sorry, I couldn't find any matching products in {shop_name}."
    # Build a natural reply listing products
    prods_text = "\n".join(
        f"{i+1}. {p['name']} – ₹{p['price']} (Color: {p['color']})"
        for i, p in enumerate(products)
    )
    prompt = (
        f"Shop name: {shop_name}\n"
        f"User query: {user_query}\n"
        f"Matching products:\n{prods_text}\n"
        f"Reply to the user in a friendly, helpful tone, in {language}."
    )
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful textile shopping assistant."},
                  {"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()
