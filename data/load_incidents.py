"""
Load incident data from CSV into PostgreSQL incidents table.

Requirements covered:
- NaN cleaned to None for SQL insertion
- Columns standardized to incidents schema
- Batch insertion with psycopg2.extras.execute_values
- DB config from environment variables
- Duplicate-safe inserts via ON CONFLICT (number) DO NOTHING
"""

from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CSV_PATH = os.getenv("INCIDENT_CSV_PATH", "data/incident_event_log.csv")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))

INCIDENT_COLUMNS: List[str] = [
    "number",
    "incident_state",
    "active",
    "reassignment_count",
    "reopen_count",
    "sys_mod_count",
    "made_sla",
    "contact_type",
    "location",
    "category",
    "subcategory",
    "u_symptom",
    "impact",
    "urgency",
    "priority",
    "assignment_group",
    "knowledge",
    "notify",
    "closed_code",
    "opened_at",
    "resolved_at",
    "closed_at",
]

TIMESTAMP_COLUMNS = ["opened_at", "resolved_at", "closed_at"]
INT_COLUMNS = ["reassignment_count", "reopen_count", "sys_mod_count"]
BOOL_COLUMNS = ["active", "made_sla", "knowledge"]

BOOL_MAP = {
    "true": True,
    "false": False,
    "1": True,
    "0": False,
    "yes": True,
    "no": False,
    "y": True,
    "n": False,
}


# ──────────────────────────────────────────────────────────────
# Database Connection
# ──────────────────────────────────────────────────────────────

def get_db_connection():
    required_vars = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


# ──────────────────────────────────────────────────────────────
# Data Cleaning
# ──────────────────────────────────────────────────────────────

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def normalize_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    key = str(value).strip().lower()
    return BOOL_MAP.get(key, None)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_column_names(df)

    missing_cols = [c for c in INCIDENT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    df = df[INCIDENT_COLUMNS].copy()

    # Convert timestamps
    for col in TIMESTAMP_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Convert integers
    for col in INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Convert booleans
    for col in BOOL_COLUMNS:
        df[col] = df[col].map(normalize_bool)

    # Ensure incident number exists (without converting nulls to string "nan")
    df["number"] = df["number"].where(df["number"].notna(), None)
    df["number"] = df["number"].map(lambda v: str(v).strip() if v is not None else None)
    df = df[df["number"].notna() & (df["number"] != "")]

    return df


# ──────────────────────────────────────────────────────────────
# Batch Insert
# ──────────────────────────────────────────────────────────────

def rows_in_batches(df: pd.DataFrame, batch_size: int) -> Iterable[List[Tuple]]:
    total = len(df)
    for i in range(0, total, batch_size):
        chunk = df.iloc[i:i + batch_size]
        yield [tuple(to_python_value(v) for v in row) for row in chunk.itertuples(index=False, name=None)]


def to_python_value(value):
    # Handles NaN, NaT, and pd.NA uniformly.
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value


def insert_incidents(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    insert_sql = f"""
        INSERT INTO incidents ({", ".join(INCIDENT_COLUMNS)})
        VALUES %s
        ON CONFLICT (number) DO NOTHING
    """

    total_inserted = 0

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for batch in rows_in_batches(df, BATCH_SIZE):
                execute_values(cur, insert_sql, batch, page_size=BATCH_SIZE)
                total_inserted += cur.rowcount if cur.rowcount > 0 else 0

        conn.commit()

    return total_inserted


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print(f"Loading CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Rows read: {len(df)}")

    cleaned = clean_dataframe(df)
    print(f"Rows after cleaning: {len(cleaned)}")

    inserted = insert_incidents(cleaned)
    skipped = len(cleaned) - inserted

    print(f"Inserted rows: {inserted}")
    print(f"Skipped rows (duplicates/conflicts): {skipped}")


if __name__ == "__main__":
    main()
