import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os

load_dotenv()

def create_occasions_table(conn, cursor):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS public.occasions (
        id SERIAL PRIMARY KEY,
        name TEXT UNIQUE NOT NULL
    );
    """
    cursor.execute(create_table_query)
    conn.commit()
    print("Table 'occasions' ensured to exist.")

def insert_default_occasions(conn, cursor):
    default_occasions = ['Wedding', 'Casual', 'Party']
    insert_query = """
    INSERT INTO public.occasions (name) VALUES (%s)
    ON CONFLICT (name) DO NOTHING;
    """
    for occasion in default_occasions:
        cursor.execute(insert_query, (occasion,))
    conn.commit()
    print(f"Inserted default occasions: {', '.join(default_occasions)}")

def main():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('dbname'),
            user=os.getenv('user'),
            password=os.getenv('password'),
            host=os.getenv('host'),
            port=os.getenv('port')
        )
        cursor = conn.cursor()

        create_occasions_table(conn, cursor)
        insert_default_occasions(conn, cursor)

        cursor.close()
        conn.close()
        print("Database connection closed.")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
