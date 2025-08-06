import asyncio
from app.core.product_search import search_products_by_image

async def main():
    image_url = "https://d2ki7eiqd260sq.cloudfront.net/Dark-Red-Pure-Organza-Banarasi-Saree3326c8f7-e149-4dc0-8d33-0bdb667ed43e.jpg"
    results = await search_products_by_image(image_url, top_k=5)
    print('results...........', results)
    for i, prod in enumerate(results):
        print(f"{i+1}. {prod['product_name']} (score={prod['score']:.4f}) → {prod['image_url']}")

if __name__ == "__main__":
    asyncio.run(main())
