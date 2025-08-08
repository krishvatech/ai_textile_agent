
import logging
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("GPT_API_KEY"))

def extract_dynamic_attributes(txt):
    """
    Function to dynamically extract color and fabric from a given sentence using GPT-3/4.
    """
    # Initialize the OpenAI client

    
    prompt = f"Extract color and fabric from the following sentence:\n\n{txt}\n\nReturn in the format: {{'color': '...', 'fabric': '...'}}"

    try:
        # Send the request to OpenAI API (GPT-3 or GPT-4)
        response = client.chat.completions.create(
            model="gpt-5-mini",  # Use GPT-4 or another compatible model
            messages=[
                {"role": "system", "content": "You are an assistant that extracts colors and fabrics from sentences."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the output from the response
        output = response.choices[0].message.content.strip()

        # Parse the output into a dictionary
        attributes = eval(output)  # Safely convert the returned string to a dictionary
        return attributes
    except Exception as e:
        logging.error(f"Error during GPT-based attribute extraction: {e}")
        return {}

# Example usage
txt = "I want a red cotton saree."
attributes = extract_dynamic_attributes(txt)
print(attributes)  # Expected output: {'color': 'red', 'fabric': 'cotton'}
