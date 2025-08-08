import csv
import os
import requests
import threading
import time
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from datetime import datetime
import base64
import logging
import re  # For number validation

load_dotenv()

# Load credentials and config from environment
EXOTEL_SID = os.getenv("EXOTEL_SID")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_API_TOKEN = os.getenv("EXOTEL_TOKEN")
EXOTEL_CALLER_ID = os.getenv("EXOPHONE")  # Your Exotel number
EXOTEL_SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN")
APP_ID = os.getenv("EXOTEL_APP_ID")

# Enhanced checks for all required vars
required_vars = {
    "EXOTEL_SID": EXOTEL_SID,
    "EXOTEL_API_KEY": EXOTEL_API_KEY,
    "EXOTEL_API_TOKEN": EXOTEL_API_TOKEN,
    "EXOTEL_CALLER_ID": EXOTEL_CALLER_ID,
    "APP_ID": APP_ID
}
for var_name, var_value in required_vars.items():
    if var_value is None:
        raise ValueError(f"{var_name} environment variable is missing")

BASE_URL = f"https://api.exotel.com/v1/Accounts/{EXOTEL_SID}/Calls/connect.json"
EXOTEL_URL = f"https://my.exotel.com/{EXOTEL_SID}/exoml/start_voice/{APP_ID}"

retry_lock = threading.Lock()
MAX_RETRIES = 2
RETRY_DELAY_BASE = 10  # Base delay (in seconds) between retries
RETRY_DELAY_MAX = 600  # Max delay between retries

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            next(csv_reader)  # Skip header if present
            for row in csv_reader:
                number = row[0].strip()
                # Validate: 10-digit Indian number
                if re.match(r'^\d{10}$', number):
                    contact_numbers.append(number)
                else:
                    logger.warning(f"Invalid number skipped: {number}")
    except Exception as e:
        logger.error(f"❌ Error reading CSV file: {e}")
    return contact_numbers

def make_outbound_call(to_number: str, is_retry=False):
    """
    Makes an outbound call using Exotel API without saving to the database.
    """
    call_sid = ""
    if not to_number.startswith("+91"):
        to_number = "+91" + to_number.lstrip("0")
    
    # Simplified payload: Standard single-leg outbound to customer
    payload = {
        "From": to_number,  # Customer's number
        "CallerId": EXOTEL_CALLER_ID,  # Your ExoPhone
        "Url": EXOTEL_URL,  # Your endpoint for XML
        "CustomField": f"outbound|{to_number}"
    }
    logger.info(f"Payload (confirm no 'To'): {payload}") 
    # Base64 encode Exotel credentials
    credentials = f"{EXOTEL_API_KEY}:{EXOTEL_API_TOKEN}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "accept": "application/json"
    }
    logger.info(f"Ngrok URL: {EXOTEL_URL} - test manually with curl")
    # Make the request with retries
    status = "failed"
    attempts = 0
    while attempts <= MAX_RETRIES:
        response = requests.post(BASE_URL, data=payload, headers=headers)
        logger.info(f"📡 Status Code: {response.status_code}")
        logger.info(f"Response Text: {response.text}")  # Full debugging

        if response.status_code == 200:
            try:
                response_json = response.json()
                call_sid = response_json.get("Call", {}).get("Sid", "")
                if call_sid:
                    # Poll status with shorter interval
                    poll_status = "in-progress"
                    poll_attempts = 0
                    max_polls = 24  # ~2 minutes max
                    while poll_status == "in-progress" and poll_attempts < max_polls:
                        time.sleep(5)  # Check every 5s
                        call_detail_url = f"https://{EXOTEL_API_KEY}:{EXOTEL_API_TOKEN}@api.exotel.com/v1/Accounts/{EXOTEL_SID}/Calls/{call_sid}.xml"
                        detail_response = requests.get(call_detail_url)
                        if detail_response.status_code == 200:
                            # logger.info(f"Detail Response: {detail_response.text}")
                            detail_root = ET.fromstring(detail_response.text)
                            status_element = detail_root.find(".//Status")
                            poll_status = status_element.text if status_element is not None else "failed"
                            logger.info(f"📞 Polling Status: {poll_status}")
                        poll_attempts += 1
                    status = poll_status
                    logger.info(f"📞 Final Call Status: {status}")
                    logger.info(f"📞 Call SID: {call_sid}")
                    # Check for short duration warning
                    if status == "completed":
                        duration_element = detail_root.find(".//Duration")
                        duration = int(duration_element.text) if duration_element is not None and duration_element.text else 0
                        if duration < 10:
                            logger.warning("⚠️ Call completed but duration too short - possible XML not executed")
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                status = "failed"
            break  # Success, exit retry loop
        else:
            logger.error("❌ Exotel API call failed. Possibly bad credentials or wrong URL.")

        attempts += 1
        if attempts <= MAX_RETRIES:
            delay = min(RETRY_DELAY_BASE * (2 ** (attempts - 1)), RETRY_DELAY_MAX)
            logger.info(f"Retry {attempts}/{MAX_RETRIES} after {delay}s")
            time.sleep(delay)

    logger.info(f"final status: {status}")
    return status, response.text if 'response' in locals() else ""

def initiate_calls_from_csv(csv_file_path):
    contact_numbers = read_contact_numbers_from_csv(csv_file_path)
    
    threads = []
    for number in contact_numbers:
        logger.info(f"Scheduling call to {number}...")
        t = threading.Thread(target=make_outbound_call, args=(number,))
        t.start()
        threads.append(t)
    
    # Wait for all calls to finish
    for t in threads:
        t.join()
        
# Call the function to start the outbound calling process
csv_file_path = "contacts.csv"  # Provide the correct path to your CSV file
initiate_calls_from_csv(csv_file_path)
