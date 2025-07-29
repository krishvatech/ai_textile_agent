import pandas as pd
import asyncio
from app.db.models import Product
from app.db.session import SessionLocal

async def import_products(tenant_id, csv_path):
    df = pd.read_csv(csv_path)
    async with SessionLocal() as session:
        for _, row in df.iterrows():
            prod = Product(
                tenant_id=tenant_id,
                name=row['name'],
                description=row.get('description', ''),
                color=row.get('color', ''),
                price=row.get('price', 0.0),
                is_rental=row.get('is_rental', False),
                available_stock=row.get('available_stock', 0),
                image_url=row.get('image_url', ''),
            )
            session.add(prod)
        await session.commit()
        print(f"Imported {len(df)} products.")

if __name__ == "__main__":
    tenant_id = int(input("Tenant ID? "))
    csv_path = input("Path to products.csv? ")
    asyncio.run(import_products(tenant_id, csv_path))
