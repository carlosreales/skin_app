import sqlite3
from datetime import datetime


DB_NAME = "skin_ai.db"


def get_connection():
    connection = sqlite3.connect(DB_NAME)
    return connection


def create_tables():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS skin_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_date TEXT,
            predicted_class TEXT,
            confidence REAL,
            image_path TEXT,
            severity TEXT,
            description TEXT
        )
    """)

    cursor.execute("PRAGMA table_info(skin_analysis)")
    columns = [column[1] for column in cursor.fetchall()]

    if "image_path" not in columns:
        cursor.execute("ALTER TABLE skin_analysis ADD COLUMN image_path TEXT")

    if "severity" not in columns:
        cursor.execute("ALTER TABLE skin_analysis ADD COLUMN severity TEXT")

    if "description" not in columns:
        cursor.execute("ALTER TABLE skin_analysis ADD COLUMN description TEXT")

    connection.commit()
    connection.close()


def save_analysis(predicted_class, confidence, image_path, severity, description):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO skin_analysis (
            analysis_date,
            predicted_class,
            confidence,
            image_path,
            severity,
            description
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        predicted_class,
        confidence,
        image_path,
        severity,
        description
    ))

    connection.commit()
    connection.close()


def get_analysis_history():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT 
            id,
            analysis_date,
            predicted_class,
            confidence,
            image_path,
            severity,
            description
        FROM skin_analysis
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    connection.close()

    return rows

def delete_all_analysis():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("DELETE FROM skin_analysis")

    connection.commit()
    connection.close()