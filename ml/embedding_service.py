import os
import time
import psycopg2
import psycopg2.extras
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT"),
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # how many incidents to embed at once


def get_db_connection():
    """Create and return a database connection."""
    return psycopg2.connect(**DB_CONFIG)


def build_incident_text(incident):
    """
    Use real text fields for much better embeddings!
    """
    parts = [
        f"Issue: {incident['initial_message'] or 'unknown'}",
        f"Type: {incident['issue_type'] or 'unknown'}",
        f"Product Area: {incident['product_area'] or 'unknown'}",
        f"Priority: {incident['priority'] or 'unknown'}",
        f"Resolution: {incident['resolution_summary'] or 'unknown'}",
        f"Platform: {incident['platform'] or 'unknown'}",
    ]
    return " | ".join(parts)


def fetch_incidents_without_embeddings(conn, limit=1000):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT i.id, i.initial_message, i.issue_type,
                   i.product_area, i.priority,
                   i.resolution_summary, i.platform
            FROM incidents i
            LEFT JOIN incident_embeddings ie ON i.id = ie.incident_id
            WHERE ie.id IS NULL
            LIMIT %s
        """, (limit,))
        return cur.fetchall()


def generate_embeddings(texts):
    """
    Call OpenAI API to generate embeddings for a list of texts.
    Returns list of embedding vectors.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def store_embeddings(conn, incident_ids, embeddings):
    """
    Store embeddings in the incident_embeddings table.
    Converts embeddings to pgvector string format.
    """
    rows = [(inc_id, str(emb)) for inc_id, emb in zip(incident_ids, embeddings)]

    if not rows:
        return

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
                INSERT INTO incident_embeddings
                    (incident_id, embedding_vector)
                VALUES %s
                ON CONFLICT DO NOTHING
            """,
            rows,
            page_size=len(rows),
        )
    conn.commit()


def run_embedding_pipeline(limit=1000):
    """
    Main pipeline:
    1. Fetch incidents without embeddings
    2. Build text for each incident
    3. Generate embeddings in batches
    4. Store in database
    """
    print(f"Starting embedding pipeline (limit: {limit} incidents)...")

    conn = get_db_connection()

    incidents = fetch_incidents_without_embeddings(conn, limit=limit)
    print(f"Found {len(incidents)} incidents without embeddings")

    if not incidents:
        print("All incidents already have embeddings!")
        conn.close()
        return

    total_embedded = 0

    for i in range(0, len(incidents), BATCH_SIZE):
        batch = incidents[i:i + BATCH_SIZE]

        texts = [build_incident_text(inc) for inc in batch]
        incident_ids = [inc['id'] for inc in batch]

        print(f"Embedding batch {i//BATCH_SIZE + 1} "
              f"({len(batch)} incidents)...", end=" ")

        try:
    
            embeddings = generate_embeddings(texts)

   
            store_embeddings(conn, incident_ids, embeddings)

            total_embedded += len(batch)
            print(f"Done. Total embedded so far: {total_embedded}")

            time.sleep(0.5)

        except Exception as e:
            print(f"Error on batch {i//BATCH_SIZE + 1}: {e}")
            conn.rollback()
            continue

    conn.close()
    print(f"\nEmbedding pipeline complete!")
    print(f"Total incidents embedded: {total_embedded}")


if __name__ == "__main__":
    run_embedding_pipeline(limit=100000)