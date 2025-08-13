from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Customer, ChatSession

async def get_or_create_customer(
    db: AsyncSession,
    tenant_id: int,
    phone: Optional[str] = None,
    whatsapp_id: Optional[str] = None,
    name: Optional[str] = None,
    preferred_language: Optional[str] = None,
) -> Customer:
    stmt = select(Customer).where(Customer.tenant_id == tenant_id)
    if phone:
        stmt = stmt.where(Customer.phone == phone)
    elif whatsapp_id:
        stmt = stmt.where(Customer.whatsapp_id == whatsapp_id)
    else:
        raise ValueError("Either phone or whatsapp_id is required.")

    result = await db.execute(stmt)
    customer = result.scalars().first()

    if not customer:
        customer = Customer(
            tenant_id=tenant_id,
            phone=phone,
            whatsapp_id=whatsapp_id,
            name=name,
            preferred_language=preferred_language or "en",
            meta_data={},
            is_active=True,
        )
        db.add(customer)
        await db.flush()
    else:
        dirty = False
        if name and not customer.name:
            customer.name = name
            dirty = True
        if preferred_language and customer.preferred_language != preferred_language:
            customer.preferred_language = preferred_language
            dirty = True
        if dirty:
            db.add(customer)
            await db.flush()
    return customer


async def get_or_open_active_session(db: AsyncSession, customer_id: int) -> ChatSession:
    stmt = (
        select(ChatSession)
        .where(ChatSession.customer_id == customer_id)
        .where(ChatSession.ended_at.is_(None))
        .order_by(desc(ChatSession.started_at))
        .limit(1)
    )
    result = await db.execute(stmt)
    session = result.scalars().first()
    if not session:
        session = ChatSession(customer_id=customer_id, started_at=datetime.utcnow(), transcript=[])
        db.add(session)
        await db.flush()
    return session


async def append_transcript_message(
    db: AsyncSession,
    chat_session: ChatSession,
    role: str,                     # "user" | "assistant" | "system"
    text: str,
    msg_id: Optional[str] = None,
    ts: Optional[datetime] = None,
    direction: Optional[str] = None,   # "in" | "out" (optional)
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    entry = {
        "role": role,
        "text": text,
        "msg_id": msg_id,
        "ts": (ts or datetime.utcnow()).isoformat(),
    }
    if direction:
        entry["direction"] = direction
    if meta:
        entry["meta"] = meta

    transcript: List[dict] = list(chat_session.transcript or [])
    # de-dup by msg_id if provided
    if entry.get("msg_id") and any(m.get("msg_id") == entry["msg_id"] for m in transcript):
        return

    transcript.append(entry)
    chat_session.transcript = transcript
    db.add(chat_session)
    await db.flush()