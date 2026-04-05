"""
FastAPI Dependencies
Database connection, dependency injection
"""

import os
import psycopg2
from contextlib import contextmanager
from dotenv import load_dotenv
from fastapi import HTTPException, status

load_dotenv()

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "decisionlens_db"),
    "user": os.getenv("DB_USER", "decisionlens"),
    "password": os.getenv("DB_PASSWORD", "decisionlens123"),
}


def get_db_connection():
    """
    Create database connection.
    Used as FastAPI dependency for endpoints.
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    except psycopg2.Error as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection failed: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@contextmanager
def get_db_connection_context():
    """
    Context manager for database connection.
    Used in background tasks and non-endpoint functions.
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    finally:
        if conn:
            conn.close()


def verify_openai_key():
    """Verify OpenAI API key is configured"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY not configured"
        )
    return api_key
