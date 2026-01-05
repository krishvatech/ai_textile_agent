import asyncio
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import select
from app.db import engine, SessionLocal
from app.db.models import Base, Tenant, User, UserRole

async def main():
    # 1) Create tables from models
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 2) Seed tenant + users (only if empty)
    async with SessionLocal() as session:
        tenant = (await session.execute(select(Tenant).limit(1))).scalars().first()
        if not tenant:
            tenant = Tenant(name="Local Shop", whatsapp_number="9999999999", language="en")
            session.add(tenant)
            await session.flush()  # gets tenant.id

        # Superadmin user
        admin = (await session.execute(select(User).where(User.email == "admin@local.com"))).scalars().first()
        if not admin:
            session.add(User(
                email="admin@local.com",
                hashed_password="admin123",   # plain text in your login code
                role=UserRole.superadmin,
                tenant_id=None,
                is_active=True
            ))

        # Tenant admin user
        tadmin = (await session.execute(select(User).where(User.email == "tenant@local.com"))).scalars().first()
        if not tadmin:
            session.add(User(
                email="tenant@local.com",
                hashed_password="tenant123",  # plain text
                role=UserRole.tenant_admin,
                tenant_id=tenant.id,
                is_active=True
            ))

        await session.commit()

    print("✅ DB ready")
    print("Login:")
    print("  superadmin -> admin@local.com / admin123")
    print("  tenant_admin -> tenant@local.com / tenant123")

if __name__ == "__main__":
    asyncio.run(main())
