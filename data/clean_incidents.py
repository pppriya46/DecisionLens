import pandas as pd
import os

CSV_PATH    = "data/data.csv"
OUTPUT_PATH = "data/incidents_cleaned.csv"

def main():
    print(f"Reading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows: {len(df)}")

    # ── Fix data types ──────────────────────────────────────
    # Parse timestamps
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

    # Fix boolean columns
    df['reopened']       = df['reopened'].astype(bool)
    df['has_attachment'] = df['has_attachment'].astype(bool)

    # Fix numeric columns
    df['resolution_time_hours'] = pd.to_numeric(
        df['resolution_time_hours'], errors='coerce'
    )
    df['csat_score'] = pd.to_numeric(df['csat_score'], errors='coerce')

    # ── Fix data quality ────────────────────────────────────
    # Replace empty strings with None
    string_cols = [
        'customer_segment', 'channel', 'product_area', 'issue_type',
        'priority', 'status', 'sla_plan', 'initial_message',
        'agent_first_reply', 'resolution_summary', 'customer_sentiment',
        'platform', 'region'
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].replace('', None)
            df[col] = df[col].replace('?', None)

    # ── Remove duplicates ───────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=['ticket_id'])
    after = len(df)
    print(f"Removed {before - after} duplicate tickets")

    # ── Show stats ──────────────────────────────────────────
    print(f"\nRows after cleaning: {len(df)}")
    print(f"\nIssue type distribution:")
    print(df['issue_type'].value_counts())
    print(f"\nPriority distribution:")
    print(df['priority'].value_counts())
    print(f"\nStatus distribution:")
    print(df['status'].value_counts())

    # Save cleaned CSV
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned CSV to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()