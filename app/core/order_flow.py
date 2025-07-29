from app.db.models import Order, Product, Customer, OrderStatus
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

async def create_order(db: AsyncSession, tenant_id, customer_id, product_id, order_type="purchase"):
    product = await db.get(Product, product_id)
    if not product:
        raise ValueError("Product not found")
    order = Order(
        tenant_id=tenant_id,
        customer_id=customer_id,
        product_id=product_id,
        order_type=order_type,
        status=OrderStatus.placed,
        price=product.price,
        created_at=datetime.utcnow()
    )
    db.add(order)
    await db.commit()
    await db.refresh(order)
    return order
