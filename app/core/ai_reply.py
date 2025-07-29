import openai
import os
openai.api_key = os.getenv("GPT_API_KEY")

def generate_reply(user_query, products, shop_name, action=None, language="en"):
    prompt = f"Shop name: {shop_name}\nUser query: {user_query}\nLanguage: {language}\n"
    if not products:
        prompt += f"No matching products. Kindly reply to the user in {language} and offer help with another search."
    else:
        prods = "\n".join(
            f"{i+1}. {p['name']} – ₹{p['price']} (Color: {p['color']})"
            for i, p in enumerate(products)
        )
        prompt += f"Matching products:\n{prods}\n"
        prompt += f"Reply in a friendly, helpful, and human-like tone. Use {language}."
        if action == "order":
            prompt += "\nUser wants to order. Ask for delivery address."
        elif action == "rental":
            prompt += "\nUser wants to rent. Ask for rental dates."
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful, polite textile sales assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content.strip()
