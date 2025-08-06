from datetime import datetime
from app.db.db_connection import get_db_connection, close_db_connection

def insert_tenant():
    conn, cursor = get_db_connection()

    print("Enter Tenant details:")
    name = input("Tenant name (unique): ").strip()
    whatsapp_number = input("WhatsApp number (unique): ").strip()
    phone_number = input("Phone number (optional): ").strip() or None
    address = input("Address (optional): ").strip() or None
    language = input("Language (default 'en'): ").strip() or "en"
    is_active_input = input("Is active? (yes/no, default yes): ").strip().lower()
    is_active = True if is_active_input in ['', 'yes', 'y'] else False

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    insert_query = '''
    INSERT INTO tenants (name, whatsapp_number, phone_number, address, language, created_at, updated_at, is_active)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    '''

    cursor.execute(insert_query, (
        name,
        whatsapp_number,
        phone_number,
        address,
        language,
        now,
        now,
        is_active
    ))
    conn.commit()

    print("Tenant inserted successfully.")

    close_db_connection(conn, cursor)


if __name__ == '__main__':
    insert_tenant()
