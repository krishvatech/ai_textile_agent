import re

def normalize_size(transcript: str) -> str | None:
    text = transcript.lower().strip().replace('.', '')
    
    # Handle double XL variations including 'double ex' and 'double excel'
    if re.search(r'double\s*(x{1,3}l|ex|excel)', text) or re.search(r'extra\s*extra\s*large', text):
        return "XXL"
    
    if re.search(r'extra\s*large|excel', text):
        return "XL"
    
    if re.search(r'large', text):
        return "L"
    
    if re.search(r'medium|med', text):
        return "M"
    
    if re.search(r'small|sm', text):
        return "S"
    
    if re.search(r'extra\s*small|xs', text):
        return "XS"
    
    if re.search(r'double\s*extra\s*small|xxs', text):
        return "XXS"

    sizes = ['xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxxl']
    if text in sizes:
        return text.upper()

    return None

def main():
    print("Type a size transcript (or 'exit' to quit):")
    while True:
        user_input = input(">> ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        normalized = normalize_size(user_input)
        if normalized is None:
            print(f"Input: '{user_input}' → Size NOT recognized, please clarify.")
        else:
            print(f"Input: '{user_input}' → Normalized size: {normalized}")

if __name__ == "__main__":
    main()
