"""
Pre-process incident_event_log.csv
Keep only the LATEST state per incident number
before loading into PostgreSQL
"""

import pandas as pd
import os

CSV_PATH     = "data/incident_event_log.csv"
OUTPUT_PATH  = "data/incidents_cleaned.csv"

# Priority order for incident states
# Higher number = more final state
STATE_PRIORITY = {
    "New":        1,
    "Active":     2,
    "Awaiting User Info": 3,
    "Awaiting Problem":   4,
    "Awaiting Vendor":    5,
    "Resolved":   6,
    "Closed":     7,
}

def get_state_priority(state):
    """Return priority number for a given state"""
    return STATE_PRIORITY.get(str(state).strip(), 0)

def main():
    print(f"Reading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows (all events): {len(df)}")
    print(f"Unique incidents       : {df['number'].nunique()}")

    # Add priority column based on state
    df['state_priority'] = df['incident_state'].map(get_state_priority)

    # Sort by incident number + state priority
    # So the most final state is LAST for each incident
    df = df.sort_values(
        ['number', 'state_priority'],
        ascending=[True, True]
    )

    # Keep only the LAST (most final) row per incident
    df_final = df.groupby('number').last().reset_index()

    # Drop the helper column
    df_final = df_final.drop(columns=['state_priority'])

    print(f"Rows after deduplication: {len(df_final)}")
    print(f"\nState distribution:")
    print(df_final['incident_state'].value_counts())

    # Save cleaned CSV
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned CSV to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()