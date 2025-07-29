
async def handle_confirmation(user_id, user_message, db, tenant_id):
    # Save confirmation/order info, ask for delivery address, payment method, etc.
    # Move to follow-up/feedback phase after order
    return "followup"
