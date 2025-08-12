# app/db/rental_utils.py
from datetime import date, datetime, time
from typing import Optional
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Rental, RentalStatus

def _day_bounds(d: date):
    return datetime.combine(d, time.min), datetime.combine(d, time.max)

async def is_variant_available(
    db: AsyncSession,
    variant_id: int,
    start_date: date,
    end_date: Optional[date] = None,
) -> bool:
    """
    Available (True) unless there's an OVERLAPPING rental with status == ACTIVE.
    Rentals with status RETURNED or CANCELLED DO NOT block availability.
    """
    if end_date is None:
        end_date = start_date
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    sdt, _ = _day_bounds(start_date)
    _, edt2 = _day_bounds(end_date)

    q = (
        select(Rental.id)
        .where(
            Rental.product_variant_id == variant_id,
            Rental.status == RentalStatus.active,               # only ACTIVE blocks
            Rental.rental_start_date <= edt2,                   # overlap check
            or_(Rental.rental_end_date.is_(None), Rental.rental_end_date >= sdt),
        )
        .limit(1)
    )
    row = (await db.execute(q)).first()
    return row is None