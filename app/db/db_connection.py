import psycopg2


def get_db_connection():
    """Establish and return a new database connection and cursor."""
    conn = psycopg2.connect(
        dbname='textile-agent',
        user='postgres',
        password='textileagent22',
        host='textile-agent.ctwgo22okz6g.ap-south-1.rds.amazonaws.com',
        port='5432'
    )
    cursor = conn.cursor()
    return conn, cursor


def close_db_connection(conn, cursor):
    """Close the cursor and connection."""
    cursor.close()
    conn.close()
