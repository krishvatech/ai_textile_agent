import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
    
client = OpenAI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

def detect_language(text):
    # Quick and dirty! Replace with your own detector if needed
    guj_chars = set("અઆઇઈઉઊઋએઐઓઔકખગઘઙચછજઝઞટઠડઢણતથદધનપફબભમયરલવશષસહળાાંઇી") 
    if any(c in guj_chars for c in text):
        return "gu"
    return "en"

def get_query_embedding(text):
    resp = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return resp.data[0].embedding

def query_pinecone(query_embedding, top_k=2):
    result = index.query(vector=query_embedding, top_k=5, include_metadata=True,namespace="ns1")
    print(result)
    products = []
    for match in result.matches:
        meta = match.metadata
        products.append({
            "id": meta.get("id"),
            "category": meta.get("category"),
            "color": meta.get("color"),
            "description": meta.get("description"),
            "fabric": meta.get("fabric"),
            "price": meta.get("price"),
            "name": meta.get("product_name"),
            "size": meta.get("size"),
        })
    return products
def generate_reply(user_query, products, language, shop_name="Krishna Textiles", intent="product_search"):
    prompt = f"""
You are a polite, expert textile shop assistant from India.
The user's detected intent is: {intent}.
Always reply in the {language} language.
Keep replies short—no longer than 2-3 WhatsApp message lines, max 2 sentences.
Start with the main point and only add ONE clarifying detail or follow-up question.
If the intent is 'product_search', confirm if the product is available and ask for fabric/design preference.
If you need more info, request it briefly and politely in the same language.
User: "{user_query}"
Assistant:
"""
    if not products:
        prompt += f"No matching products found.\n"
    else:
        # For English response
        if language == "en":
            prods = "\n".join(
                f"{i+1}. {p.get('name') or p.get('product_name', 'Product')} – ₹{p.get('price', '?')} (Color: {p.get('color', '?')}, Fabric: {p.get('fabric', '?')})"
                for i, p in enumerate(products)
            )
            prompt += f"Matching products:\n{prods}\n"
        
        # For Gujarati response
        elif language == "gu":
            prods = "\n".join(
                f"{i+1}. {p.get('name') or p.get('product_name', 'Product')} – ₹{p.get('price', '?')} (રંગ: {p.get('color', '?')}, ફેબ્રિક: {p.get('fabric', '?')})"
                for i, p in enumerate(products)
            )
            prompt += f"મેળવેલ ઉત્પાદનો:\n{prods}\n"

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "system",
            "content": (
                "You are a warm, polite, and chatty textile shop owner who loves helping and chatting with customers, "
                "whether it's about shopping or just casually talking."
            )
        },
        {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()


def main():
    user_query = input("User transcript: ").strip()
    language = detect_language(user_query)
    print(f"Detected language: {language}")

    embedding = get_query_embedding(user_query)
    print(f"Embedding vector length: {len(embedding)}")
    
    products = query_pinecone(embedding, top_k=2)
    print(f"Found {len(products)} products.")

    # Use asyncio to run the async function in sync context
    import asyncio
    answer = generate_reply(
        user_query=user_query,
        products=products,
        language=language,
        shop_name="Krishna Textiles",
        intent="product_search"
    )
    print("\nBot reply:\n", answer)

if __name__ == "__main__":
    main()
