# ml/embedding_service.py
# Purpose: Generate embeddings for all incidents and store in pgvector
# This is Layer 1 of our custom RAG pipeline

import os
import time
import psycopg2
import psycopg2.extras
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Setup ──────────────────────────────────────────────────────────────
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
    Combine incident fields into one string for embedding.
    The richer the text, the better the similarity search.
    """
    parts = [
        f"Category: {incident['category'] or 'unknown'}",
        f"Subcategory: {incident['subcategory'] or 'unknown'}",
        f"Symptom: {incident['u_symptom'] or 'unknown'}",
        f"Priority: {incident['priority'] or 'unknown'}",
        f"Impact: {incident['impact'] or 'unknown'}",
        f"Urgency: {incident['urgency'] or 'unknown'}",
        f"Resolution: {incident['closed_code'] or 'unknown'}",
    ]
    return " | ".join(parts)


def fetch_incidents_without_embeddings(conn, limit=1000):
    """
    Fetch incidents that don't have embeddings yet.
    Uses LEFT JOIN to find gaps.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT i.id, i.category, i.subcategory, i.u_symptom,
                   i.priority, i.impact, i.urgency, i.closed_code
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
    # Convert embeddings to pgvector format (string like "[0.1, 0.2, ...]")
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

    # Fetch incidents that need embeddings
    incidents = fetch_incidents_without_embeddings(conn, limit=limit)
    print(f"Found {len(incidents)} incidents without embeddings")

    if not incidents:
        print("All incidents already have embeddings!")
        conn.close()
        return

    # Process in batches
    total_embedded = 0

    for i in range(0, len(incidents), BATCH_SIZE):
        batch = incidents[i:i + BATCH_SIZE]

        # Build text for each incident in batch
        texts = [build_incident_text(inc) for inc in batch]
        incident_ids = [inc['id'] for inc in batch]

        print(f"Embedding batch {i//BATCH_SIZE + 1} "
              f"({len(batch)} incidents)...", end=" ")

        try:
            # Generate embeddings
            embeddings = generate_embeddings(texts)

            # Store in database
            store_embeddings(conn, incident_ids, embeddings)

            total_embedded += len(batch)
            print(f"Done. Total embedded so far: {total_embedded}")

            # Small delay to be nice to the API
            time.sleep(0.5)

        except Exception as e:
            print(f"Error on batch {i//BATCH_SIZE + 1}: {e}")
            conn.rollback()
            continue

    conn.close()
    print(f"\nEmbedding pipeline complete!")
    print(f"Total incidents embedded: {total_embedded}")


if __name__ == "__main__":
    # Start safe — embed first 1000 incidents
    # Once verified, we'll run for all 24,918
    run_embedding_pipeline(limit=24918)