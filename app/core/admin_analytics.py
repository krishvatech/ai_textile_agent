from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.db.models import Order, Product
from sqlalchemy import func

router = APIRouter()

@router.get("/admin/analytics/orders/")
async def order_stats(tenant_id: int, db: AsyncSession = Depends(get_db)):
    total_orders = await db.scalar(db.query(func.count(Order.id)).filter(Order.tenant_id == tenant_id))
    rentals = await db.scalar(db.query(func.count(Order.id)).filter(Order.tenant_id == tenant_id, Order.order_type == "rental"))
    sales = await db.scalar(db.query(func.count(Order.id)).filter(Order.tenant_id == tenant_id, Order.order_type == "purchase"))
    return {"total_orders": total_orders, "rentals": rentals, "sales": sales}

@router.get("/admin/analytics/top-products/")
async def top_products(tenant_id: int, db: AsyncSession = Depends(get_db)):
    top = await db.execute(
        db.query(Product.name_en, func.count(Order.id).label("order_count"))
        .join(Order, Product.id == Order.product_id)
        .filter(Product.tenant_id == tenant_id)
        .group_by(Product.name_en)
        .order_by(func.count(Order.id).desc())
        .limit(5)
    )
    return [{"name": row[0], "orders": row[1]} for row in top.fetchall()]
