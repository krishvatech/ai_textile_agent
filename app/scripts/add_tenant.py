import asyncio
from app.db.models import Tenant, Product, Base
from app.db import engine, SessionLocal

async def main():
    # Create tables if not exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Add a new tenant
    async with SessionLocal() as session:
        name = input("Enter shop/tenant name: ")
        whatsapp = input("WhatsApp number for this shop (for routing): ")

        tenant = Tenant(name=name, whatsapp_number=whatsapp)
        session.add(tenant)
        await session.commit()
        print(f"Tenant added! ID: {tenant.id}")

        # Add sample product
        pname = input("Add a product name: ")
        pdesc = input("Description: ")
        pcolor = input("Color: ")
        pprice = float(input("Price: "))
        pimage = input("Image URL: ")
        prod = Product(
            name=pname, description=pdesc, color=pcolor,
            price=pprice, image_url=pimage, tenant_id=tenant.id
        )
        session.add(prod)
        await session.commit()
        print("Sample product added!")

if __name__ == "__main__":
    asyncio.run(main())
