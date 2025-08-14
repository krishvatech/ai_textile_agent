from typing import Dict, Any, Optional
from datetime import datetime, date

from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState
from app.agent.tools import (
    tool_product_search,
    tool_availability_filter,
    tool_rerank,
)
from app.core.lang_utils import detect_language
from app.core.intent_utils import detect_textile_intent_openai
from app.core.ai_reply import analyze_message


# --- Local text extractor (prevents dicts from becoming str(dict)) ---
def _extract_reply_text(payload) -> str:
    if isinstance(payload, dict):
        for k in ("reply_text", "reply", "text", "message", "answer"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    if isinstance(payload, str):
        return payload.strip()
    return "Thanks! How can I help you with fabrics or clothing today?"


# --- Node functions ---
async def node_detect_language(state: AgentState) -> AgentState:
    text = state.get("text", "")
    last_language = state.get("language") or "en-IN"

    # detect_language can return (lang, confidence) or just lang
    detected = await detect_language(text, last_language)
    if isinstance(detected, tuple):
        lang = detected[0] or "en-IN"
    else:
        lang = detected or "en-IN"

    state["language"] = lang
    return state


async def node_classify(state: AgentState) -> AgentState:
    # Garment keywords fallback (classifier -> "other" but text clearly product-y)
    garment_keywords = {
        "saree", "lehenga", "choli", "kurta", "sherwani", "blouse", "dupatta",
        "gown", "dress", "salwar", "suit", "chaniya", "chaniya choli",
        "pant", "trouser", "skirt", "top", "shirt", "tshirt"
    }

    intent, entities, confidence = await detect_textile_intent_openai(
        text=state.get("text", ""),
        detected_language=state.get("language", "en"),
    )
    state["intent"] = intent
    state["entities"] = entities or {}
    state["intent_confidence"] = confidence or 0.0

    if (not state["intent"] or state["intent"] == "other") and state["intent_confidence"] < 0.55:
        text_low = (state.get("text") or "").lower()
        if any(k in text_low for k in garment_keywords):
            state["intent"] = "product_search"

    return state


async def node_retrieve(state: AgentState) -> AgentState:
    intent = (state.get("intent") or "").lower()
    if intent in {"product_search", "rental_inquiry", "price_inquiry", "catalog_request"}:
        products = await tool_product_search(
            state.get("entities") or {},
            state["tenant_id"],
            state.get("text"),
        )
        state["products"] = products
    else:
        state["products"] = []
    return state


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
    ents = state.get("entities") or {}
    start = ents.get("date") or ents.get("rental_date") or ents.get("from_date")
    end = ents.get("to_date")
    d_start = _parse_date(start)
    d_end = _parse_date(end) if end else None

    # Only filter if we have a date and retrieved products (and DB session if present)
    if not d_start or not state.get("products"):
        return state

    db = state.get("db")  # Optional: state may or may not provide DB
    state["products"] = await tool_availability_filter(db, state["products"], d_start, d_end)
    return state


async def node_alternates(state: AgentState) -> AgentState:
    # If no products after availability filtering, broaden constraints once
    if state.get("products"):
        return state

    ents = dict(state.get("entities") or {})
    for key in ("size", "color", "occasion"):
        ents.pop(key, None)
    if ents.get("is_rental") is True:
        ents.pop("is_rental", None)

    new_items = await tool_product_search(ents, state["tenant_id"], state.get("text"))
    state["products"] = new_items

    # Add a small hint that we broadened results
    prev_hint = state.get("reply") or ""
    hint = "\n\n(Showing close matches and alternates.)"
    state["reply"] = (prev_hint + hint).strip() if prev_hint else ""
    return state


QUICK_SUGGESTIONS = ["Different size", "Change color", "See alternates", "Confirm order"]


async def node_rerank(state: AgentState) -> AgentState:
    state["products"] = tool_rerank(state.get("products") or [], state.get("entities") or {})
    return state


async def node_respond(state: AgentState) -> AgentState:
    reply_payload = await analyze_message(
        text=state.get("text", ""),
        tenant_id=state.get("tenant_id"),
        tenant_name=state.get("tenant_name", ""),
        language=state.get("language", "en"),
        intent=state.get("intent"),
        new_entities=state.get("entities") or {},
        intent_confidence=state.get("intent_confidence", 0.0),
        mode="chat",
    )

    # Always extract a human string; never str(dict)
    state["reply"] = _extract_reply_text(reply_payload)

    # Add quick replies footer for product flows
    if (state.get("intent") in {"product_search", "rental_inquiry", "catalog_request", "price_inquiry"}) and state.get("products"):
        footer = "\n\nQuick replies: " + " | ".join(QUICK_SUGGESTIONS)
        state["reply"] = (state["reply"] + footer) if state["reply"] else footer

    return state


# --- Graph wiring ---
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("detect_language", node_detect_language)
    g.add_node("classify", node_classify)
    g.add_node("retrieve", node_retrieve)
    g.add_node("check_availability", node_check_availability)
    g.add_node("alternates", node_alternates)
    g.add_node("rerank", node_rerank)
    g.add_node("respond", node_respond)

    g.add_edge(START, "detect_language")
    g.add_edge("detect_language", "classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "check_availability")
    g.add_edge("check_availability", "alternates")
    g.add_edge("alternates", "rerank")
    g.add_edge("rerank", "respond")
    g.add_edge("respond", END)
    return g.compile()


graph = build_graph()


# --- Public runner ---
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
