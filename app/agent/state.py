
from typing import TypedDict, Optional, List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession

class AgentState(TypedDict, total=False):
    user_id: str
    tenant_id: int
    tenant_name: str
    text: str
    language: str
    intent: str
    entities: Dict[str, Any]
    intent_confidence: float
    products: List[Dict[str, Any]]
    reply: str
    error: Optional[str]
    db: Optional[AsyncSession]