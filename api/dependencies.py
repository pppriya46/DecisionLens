"""
FastAPI Dependencies
Database connection, dependency injection
"""

import os
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from typing import Generator

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


def get_model_path(model_name: str = "severity") -> str:
    """Get model file path"""
    if model_name == "severity":
        model_path = "ml/models/severity_rf_v1.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model file not found: {model_path}"
            )
        return model_path
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unknown model: {model_name}"
    )
