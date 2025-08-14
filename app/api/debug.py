
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
from app.agent.graph import run_graph_for_text
from app.agent.llama_index_ingest import ingest_tenant_catalog

router = APIRouter()

class DebugChatIn(BaseModel):
    user_id: str
    tenant_id: int
    tenant_name: str
    text: str

@router.post("/rag_chat")
async def rag_chat(payload: DebugChatIn) -> Dict[str, Any]:
    try:
        return await run_graph_for_text(
            user_id=payload.user_id,
            tenant_id=payload.tenant_id,
            tenant_name=payload.tenant_name,
            text=payload.text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class IngestIn(BaseModel):
    tenant_id: int

@router.post("/ingest")
async def ingest(payload: IngestIn) -> Dict[str, Any]:
    try:
        return await ingest_tenant_catalog(payload.tenant_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
