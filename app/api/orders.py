from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.db.models import Order, Product, Customer, OrderStatus
from datetime import datetime
from app.utils.payments import create_payment_link

router = APIRouter()

@router.post("/order/")
async def place_order(
    tenant_id: int,
    customer_id: int,
    product_id: int,
    order_type: str = "purchase",  # or "rental"
    start_date: str = None,        # for rental
    end_date: str = None,          # for rental
    db: AsyncSession = Depends(get_db)
):
    product = await db.get(Product, product_id)
    if not product or product.tenant_id != tenant_id:
        raise HTTPException(404, "Product not found for this tenant")
    price = product.price
    order = Order(
        tenant_id=tenant_id,
        customer_id=customer_id,
        product_id=product_id,
        order_type=order_type,
        status=OrderStatus.placed,
        price=price,
        start_date=datetime.fromisoformat(start_date) if start_date else None,
        end_date=datetime.fromisoformat(end_date) if end_date else None,
    )
    db.add(order)
    await db.commit()
    await db.refresh(order)

    # --- Payment link (generate with Razorpay/Stripe)
    payment_link = await create_payment_link(order)
    order.payment_link = payment_link
    order.status = OrderStatus.placed
    await db.commit()

    return {
        "order_id": order.id,
        "status": order.status,
        "price": order.price,
        "payment_link": payment_link,
        "message": "Order created. Please pay to confirm!"
    }
