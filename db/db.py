import mysql.connector
from typing import List
import os

def get_mysql_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345678",
        database="chatbot"
    )

# def get_mysql_connection():
#     return mysql.connector.connect(
#         host=os.getenv("MYSQL_HOST", "mysql"),
#         user=os.getenv("MYSQL_USER", "root"),
#         password=os.getenv("MYSQL_PASSWORD", "12345678"),
#         database=os.getenv("MYSQL_DATABASE", "chatbot"),
#         port=int(os.getenv("MYSQL_PORT", "3306"))
#     )


def get_etiquettes() -> List[str]:
    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, description FROM etiquette")
    result = [f"{row[0]}: {row[1]}" for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return result

from typing import Dict

def get_parameters() -> Dict[str, int]:
    """
    Fetches key-value pairs from the general_configuration table.

    Returns:
        A dictionary where keys are parameter names and values are integers.
        Example: { "max_message_count": 3, "min_confidence_threshold": 75 }
    """
    conn = get_mysql_connection()
    cursor = conn.cursor()

    query = "SELECT name, value FROM general_configuration"
    cursor.execute(query)

    parameters = {name: int(value) for name, value in cursor.fetchall()}

    cursor.close()
    conn.close()

    return parameters
