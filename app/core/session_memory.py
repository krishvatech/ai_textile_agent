# app/core/session_memory.py
from collections import defaultdict

user_sessions = defaultdict(dict)

def get_session(user_id):
    return user_sessions[user_id]

def set_session(user_id, data):
    user_sessions[user_id].update(data)
