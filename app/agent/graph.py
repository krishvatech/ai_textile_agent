
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from app.agent.state import AgentState
from app.agent.tools import tool_product_search, tool_availability_filter, tool_rerank
from app.core.lang_utils import detect_language
from app.core.intent_utils import detect_textile_intent_openai
from app.core.ai_reply import analyze_message

# --- Node functions ---
async def node_detect_language(state: AgentState) -> AgentState:
    text = state.get("text", "")
    # Use previous state language if available; default to en-IN for WhatsApp flow
    last_language = state.get("language") or "en-IN"

    # detect_language is async and can return either (lang, confidence) or just lang
    detected = await detect_language(text, last_language)
    if isinstance(detected, tuple):
        lang = detected[0] or "en-IN"
    else:
        lang = detected or "en-IN"

    state["language"] = lang
    return state


async def node_classify(state: AgentState) -> AgentState:
    intent, entities, confidence = await detect_textile_intent_openai(
        text=state.get("text", ""),
        detected_language=state.get("language", "en")
    )
    state["intent"] = intent
    state["entities"] = entities or {}
    state["intent_confidence"] = confidence or 0.0
    return state

async def node_retrieve(state: AgentState) -> AgentState:
    # Retrieve only for product-search-like intents
    if (state.get("intent") or "") in {"product_search", "rental_inquiry", "price_inquiry", "catalog_request"}:
        products = await tool_product_search(state.get("entities") or {}, state["tenant_id"], state.get("text"))
        state["products"] = products
    else:
        state["products"] = []
    return state


from datetime import datetime, date
from typing import Optional
from app.core.rental_utils import is_variant_available  # referenced by tool

def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s or not isinstance(s, str):
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None

async def node_check_availability(state: AgentState) -> AgentState:
    # Only if date is present and we have products & DB
    ents = state.get("entities") or {}
    start = ents.get("date") or ents.get("rental_date") or ents.get("from_date")
    end = ents.get("to_date")
    d_start = _parse_date(start)
    d_end = _parse_date(end) if end else None
    if not d_start or not state.get("products"):
        return state
    db = state.get("db")
    state["products"] = await tool_availability_filter(db, state["products"], d_start, d_end)
    return state

async def node_rerank(state: AgentState) -> AgentState:
    state["products"] = tool_rerank(state.get("products") or [], state.get("entities") or {})
    return state
async def node_respond(state: AgentState) -> AgentState:
    # Use your existing reply generator; keep 'mode' as 'chat' for WhatsApp
    reply_payload = await analyze_message(
        text=state.get("text",""),
        tenant_id=state.get("tenant_id"),
        tenant_name=state.get("tenant_name",""),
        language=state.get("language","en"),
        intent=state.get("intent"),
        new_entities=state.get("entities") or {},
        intent_confidence=state.get("intent_confidence", 0.0),
        mode="chat"
    )
    # analyze_message returns a dict; extract display text if present
    if isinstance(reply_payload, dict) and "text" in reply_payload:
        state["reply"] = reply_payload["text"]
    else:
        state["reply"] = str(reply_payload)
    return state

# --- Build the graph ---
from app.db.session import SessionLocal

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("detect_language", node_detect_language)
    g.add_node("classify", node_classify)
    g.add_node("retrieve", node_retrieve)
    g.add_node("respond", node_respond)

    g.add_edge(START, "detect_language")
    g.add_edge("detect_language", "classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "respond")
    g.add_edge("respond", END)
    return g.compile()

graph = build_graph()

# --- Convenience runner ---
async def run_graph_for_text(user_id: str, tenant_id: int, tenant_name: str, text: str) -> Dict[str, Any]:
    init_state: AgentState = {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "tenant_name": tenant_name,
        "text": text,
    }
    out = await graph.ainvoke(init_state)
    return {
        "reply": out.get("reply"),
        "language": out.get("language"),
        "intent": out.get("intent"),
        "entities": out.get("entities"),
        "intent_confidence": out.get("intent_confidence"),
        "products": (out.get("products") or [])[:5],  # top-5 preview
    }
