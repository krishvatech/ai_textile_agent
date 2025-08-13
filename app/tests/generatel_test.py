import asyncio
import logging

from app.core.ai_reply import analyze_message, session_entities, session_memory

logging.basicConfig(level=logging.INFO)

TENANT_ID = 777
TENANT_NAME = "Demo Shop"

async def main():
    # Fresh session
    session_entities.pop(TENANT_ID, None)
    session_memory.pop(TENANT_ID, None)

    print("\n=== CASE A: Ask name BEFORE telling it ===")
    resA = await analyze_message(
        text="What's my name?",
        tenant_id=TENANT_ID,
        tenant_name=TENANT_NAME,
        language="en-IN",
        intent="other",
        new_entities={},
        intent_confidence=0.7,
        mode="chat",
    )
    print("REPLY A:\n", resA.get("reply_text"))
    print("ROUTER A:", resA.get("router"))

    print("\n=== CASE B: Tell the bot my name ===")
    resB = await analyze_message(
        text="My name is Avinash Od.",
        tenant_id=TENANT_ID,
        tenant_name=TENANT_NAME,
        language="en-IN",
        intent="other",
        new_entities={},
        intent_confidence=0.8,
        mode="chat",
    )
    print("REPLY B:\n", resB.get("reply_text"))
    print("SESSION ENTITIES NOW:", session_entities.get(TENANT_ID))

    print("\n=== CASE C: Ask again AFTER telling name ===")
    resC = await analyze_message(
        text="What's my name?",
        tenant_id=TENANT_ID,
        tenant_name=TENANT_NAME,
        language="en-IN",
        intent="other",
        new_entities={},
        intent_confidence=0.8,
        mode="chat",
    )
    print("REPLY C:\n", resC.get("reply_text"))
    print("ROUTER C:", resC.get("router"))

if __name__ == "__main__":
    asyncio.run(main())
