from app.core.nudge_checker import check_need_nudge
from app.core.tone_classifier import classify_tone
from app.core.mood_engine import get_mood

async def handle_followup(user_id, db):
    # Send automated feedback request or offer, adjust tone if user was unhappy
    last_mood = get_mood(user_id)
    tone = classify_tone(user_id)
    nudge = check_need_nudge(user_id)
    if last_mood == "frustrated":
        msg = "Sorry you didn’t have a great experience. Can we help you find something else?"
    elif nudge:
        msg = "Would you like to see new arrivals for your next event?"
    else:
        msg = "Thanks for shopping/renting with us! Your feedback helps us improve."
    return msg
