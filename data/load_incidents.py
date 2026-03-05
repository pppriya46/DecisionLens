import os
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "data/incidents_cleaned.csv"

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "decisionlens_db"),
    "user":     os.getenv("DB_USER", "decisionlens"),
    "password": os.getenv("DB_PASSWORD", "decisionlens123"),
}

def load_incidents():
    print(f"Loading CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Rows to load: {len(df)}")

    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    inserted = 0
    skipped  = 0

    for _, row in df.iterrows():
        try:
            cur.execute("""
                INSERT INTO incidents (
                    ticket_id, created_at, customer_id,
                    customer_segment, channel, product_area,
                    issue_type, priority, status, sla_plan,
                    initial_message, agent_first_reply,
                    resolution_summary, resolution_time_hours,
                    reopened, customer_sentiment, csat_score,
                    has_attachment, platform, region
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (ticket_id) DO NOTHING
            """, (
                row['ticket_id'],
                row['created_at']              if pd.notna(row['created_at'])              else None,
                row['customer_id'],
                row['customer_segment']        if pd.notna(row['customer_segment'])        else None,
                row['channel']                 if pd.notna(row['channel'])                 else None,
                row['product_area']            if pd.notna(row['product_area'])            else None,
                row['issue_type']              if pd.notna(row['issue_type'])              else None,
                row['priority']                if pd.notna(row['priority'])                else None,
                row['status']                  if pd.notna(row['status'])                  else None,
                row['sla_plan']                if pd.notna(row['sla_plan'])                else None,
                row['initial_message']         if pd.notna(row['initial_message'])         else None,
                row['agent_first_reply']       if pd.notna(row['agent_first_reply'])       else None,
                row['resolution_summary']      if pd.notna(row['resolution_summary'])      else None,
                row['resolution_time_hours']   if pd.notna(row['resolution_time_hours'])   else None,
                bool(row['reopened'])          if pd.notna(row['reopened'])                else None,
                row['customer_sentiment']      if pd.notna(row['customer_sentiment'])      else None,
                int(row['csat_score'])         if pd.notna(row['csat_score'])              else None,
                bool(row['has_attachment'])    if pd.notna(row['has_attachment'])          else None,
                row['platform']                if pd.notna(row['platform'])                else None,
                row['region']                  if pd.notna(row['region'])                  else None,
            ))
            inserted += 1

            if inserted % 1000 == 0:
                conn.commit()
                print(f"Inserted {inserted} rows...")

        except Exception as e:
            skipped += 1
            conn.rollback()
            print(f"Error on row {row['ticket_id']}: {e}")
            continue

    conn.commit()
    cur.close()
    conn.close()

    print(f"\nDone!")
    print(f"Inserted : {inserted}")
    print(f"Skipped  : {skipped}")

if __name__ == "__main__":
    load_incidents()