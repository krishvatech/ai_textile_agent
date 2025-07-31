import openai

# Function to extract color and fabric dynamically using GPT-3/4
def extract_dynamic_attributes_using_gpt(text):
    prompt = f"Extract color and fabric from the following sentence:\n\n{text}\n\nReturn in the format: {{'color': '...', 'fabric': '...'}}"
    
    # Send the request to OpenAI API (GPT-3 or GPT-4)
    response = openai.Completion.create(
        engine="gpt-4.1-mini",  # Or use "text-davinci-003" for GPT-3
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5
    )

    # Extract the output from the response
    output = response.choices[0].text.strip()

    return output

# Example usage
txt = "I want a red cotton saree."
attributes = extract_dynamic_attributes_using_gpt(txt)
print(attributes)  # Example output: {'color': 'red', 'fabric': 'cotton'}
