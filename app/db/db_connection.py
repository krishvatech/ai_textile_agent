import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

def get_db_connection():
    """Establish and return a new database connection and cursor."""
    conn = psycopg2.connect(
        dbname=os.getenv('dbname'),
        user=os.getenv('user'),
        password=os.getenv('password'),
        host=os.getenv('host'),
        port=os.getenv('port')
    )
    cursor = conn.cursor()
    return conn, cursor


def close_db_connection(conn, cursor):
    """Close the cursor and connection."""
    cursor.close()
    conn.close()
