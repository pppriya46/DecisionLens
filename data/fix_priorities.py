import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "decisionlens_db"),
    "user":     os.getenv("DB_USER", "decisionlens"),
    "password": os.getenv("DB_PASSWORD", "decisionlens123"),
}

def assign_priority(message, issue_type, sentiment, product_area):
    message      = str(message or '').lower()
    issue_type   = str(issue_type or '').lower()
    sentiment    = str(sentiment or '').lower()
    product_area = str(product_area or '').lower()

    # ── URGENT ─────────────────────────────────────────────
    # Loosened: include more keywords and allow negative sentiment to bump up
    urgent_keywords = [
        'outage', 'system down', 'not responding', 'data loss', 'breach',
        'critical', 'crash', 'cannot access', 'failure', 'all users', 'site down',
        'major', 'urgent', 'emergency', 'immediately', 'production', 'stopped'
    ]
    if any(w in message for w in urgent_keywords):
        return 'urgent'
    if issue_type in ['outage', 'security_breach', 'data_loss']:
        return 'urgent'
    if sentiment == 'very_negative' and ('not working' in message or 'down' in message):
        return 'urgent'

    # ── URGENT ─────────────────────────────────────────────
    # System wide outages, data loss, security breaches
    urgent_message = any(w in message for w in [
        'complete outage', 'system down', 'not responding',
        'data loss', 'data breach', 'security breach',
        'critical failure', 'total failure', 'production down',
        'cannot access', 'all users affected', 'site down'
    ])
    urgent_type = issue_type in ['outage', 'security_breach', 'data_loss']

    if urgent_message or urgent_type:
        return 'urgent'

    # ── HIGH ───────────────────────────────────────────────
    # Payment failures, security concerns, very negative sentiment
    high_message = any(w in message for w in [
        'payment not', 'cannot pay', 'charge failed',
        'unauthorized', 'suspicious activity', 'account locked',
        'data exposed', 'not working at all', 'completely broken',
        'lost all', 'deleted', 'corrupted'
    ])
    high_type = issue_type in ['billing', 'payment', 'security', 'account_access']
    high_area = product_area in ['billing', 'payment', 'authentication']
    high_sentiment = sentiment == 'very_negative'

    if high_message or (high_type and high_sentiment) or (high_area and high_sentiment):
        return 'high'

    if high_type or high_area:
        return 'high'

    # ── MEDIUM ─────────────────────────────────────────────
    # Bugs, errors, performance issues, feature not working
    medium_message = any(w in message for w in [
        'error', 'bug', 'slow', 'timeout', 'not working',
        'failing', 'issue', 'problem', 'incorrect', 'wrong',
        'missing', 'broken', 'not loading', 'keeps crashing'
    ])
    medium_type = issue_type in [
        'bug', 'performance', 'api_integration',
        'feature_request', 'integration'
    ]
    medium_sentiment = sentiment in ['negative', 'very_negative']

    if medium_message and medium_sentiment:
        return 'medium'

    if medium_type:
        return 'medium'

    if medium_message:
        return 'medium'

    # ── LOW ────────────────────────────────────────────────
    # General questions, how-to, positive/neutral sentiment
    return 'low'


def fix_priorities():
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    print("Loading incidents...")
    cur.execute("""
        SELECT id, initial_message, issue_type, customer_sentiment, product_area
        FROM incidents
    """)
    incidents = cur.fetchall()
    print(f"Loaded {len(incidents)} incidents")

    print("Reassigning priorities...")
    updated  = 0
    priority_counts = {'urgent': 0, 'high': 0, 'medium': 0, 'low': 0}

    for inc_id, message, issue_type, sentiment, product_area in incidents:
        new_priority = assign_priority(message, issue_type, sentiment, product_area)
        priority_counts[new_priority] += 1
        cur.execute(
            "UPDATE incidents SET priority = %s WHERE id = %s",
            (new_priority, inc_id)
        )
        updated += 1

        if updated % 10000 == 0:
            conn.commit()
            print(f"Updated {updated} incidents...")

    conn.commit()

    print("\nNew priority distribution:")
    total = sum(priority_counts.values())
    for priority, count in sorted(priority_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total * 100
        print(f"  {priority}: {count} ({pct:.1f}%)")

    cur.close()
    conn.close()
    print(f"\n✓ Done! Updated {updated} incidents")


if __name__ == "__main__":
    fix_priorities()