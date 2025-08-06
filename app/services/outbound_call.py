import csv
import os
import requests
import threading
import time
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from datetime import datetime
import base64
import threading


# Load environment variables from `.env` file
load_dotenv()

# Load credentials and config from environment
EXOTEL_SID = os.getenv("EXOTEL_SID")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_API_TOKEN = os.getenv("EXOTEL_TOKEN")
EXOTEL_CALLER_ID = os.getenv("EXOPHONE")  # Your Exotel number
EXOTEL_SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN")
APP_ID = os.getenv("EXOTEL_APP_ID")

if EXOTEL_API_KEY is None or EXOTEL_API_TOKEN is None:
    raise ValueError("EXOTEL_API_KEY or EXOTEL_API_TOKEN environment variables are missing")

BASE_URL = f"https://api.exotel.com/v1/Accounts/{EXOTEL_SID}/Calls/connect.json"
EXOTEL_URL = f"https://87f8f73836e9.ngrok-free.app/exotel/start_voice/{APP_ID}"

retry_lock = threading.Lock()
MAX_RETRIES = 2
RETRY_DELAY_BASE = 10  # Base delay (in seconds) between retries
RETRY_DELAY_MAX = 600  # Max delay between retries

def read_contact_numbers_from_csv(csv_file_path):
    """
    Reads contact numbers from a CSV file.
    :param csv_file_path: Path to the CSV file containing contact numbers.
    :return: List of contact numbers.
    """
    contact_numbers = []
    try:
        with open(csv_file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            # Assuming the CSV file has a header row
            next(csv_reader)  # Skip header if present
            for row in csv_reader:
                contact_numbers.append(row[0])  # Assuming the number is in the first column
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
    return contact_numbers

def make_outbound_call(to_number: str, is_retry=False):
    """
    Makes an outbound call using Exotel API without saving to the database.
    """
    call_sid = ""
    if not to_number.startswith("+91"):
        to_number = "+91" + to_number.lstrip("0")
    # Prepare payload for Exotel API call
    payload = {
        "From": to_number,  # Use your Exotel caller ID
        "To": EXOTEL_CALLER_ID,  # The recipient's number is the "To" number
        "CallerId": EXOTEL_CALLER_ID,
        "Url": EXOTEL_URL,
        "CustomField": f"outbound|{to_number}"  # Optionally pass some info in CustomField
    }

    # Base64 encode Exotel credentials for authentication
    credentials = f"{EXOTEL_API_KEY}:{EXOTEL_API_TOKEN}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "accept": "application/json"
    }

    # Make the request to the Exotel API
    response = requests.post(BASE_URL, data=payload, headers=headers)
    print("📡 Status Code:", response.status_code)

    status = "failed"  # ✅ always defined
    
    if response.status_code == 200:
        try:
            response_json = response.json()
            call_sid = response_json.get("Call", {}).get("Sid", "")
            if not call_sid:
                print("❌ No call SID returned. Cannot fetch call status.")
                status = "failed"
                end_time = datetime.now()
            else:
                # Wait for 65 seconds before checking the status
                time.sleep(65)
                # Fetch call details using call SID
                call_detail_url = f"https://{EXOTEL_API_KEY}:{EXOTEL_API_TOKEN}@api.exotel.com/v1/Accounts/{EXOTEL_SID}/Calls/{call_sid}"
                detail_response = requests.get(call_detail_url)
                detail_root = ET.fromstring(detail_response.text)
                status_element = detail_root.find(".//Status")
                status = status_element.text if status_element is not None else "failed"
                print("📞 Call Status from XML:", status)
                print("📞 Call SID:", call_sid)
        except Exception as e:
            print("❌ Failed to parse JSON:", e)
            status = "failed"
        end_time = datetime.now()
    else:
        print("❌ Exotel API call failed. Possibly bad credentials or wrong URL.")
        status = "failed"  # ✅ prevent crash

    print("final status:", status)
    return status, response.text


def initiate_calls_from_csv(csv_file_path):
    contact_numbers = read_contact_numbers_from_csv(csv_file_path)
    
    threads = []
    for number in contact_numbers:
        print(f"Scheduling call to {number}...")
        t = threading.Thread(target=make_outbound_call, args=(number,))
        t.start()
        threads.append(t)
    
    # Optional: wait for all calls to finish before exiting
    for t in threads:
        t.join()
        
# Call the function to start the outbound calling process
csv_file_path = "contacts.csv"  # Provide the correct path to your CSV file
initiate_calls_from_csv(csv_file_path)

