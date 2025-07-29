from langdetect import detect

def detect_language(text):
    try:
        lang = detect(text)
        if lang.startswith("en"):
            return "en"
        elif lang.startswith("gu"):
            return "gu"
        elif lang.startswith("hi"):
            return "hi"
        else:
            return "en"
    except Exception:
        return "en"
