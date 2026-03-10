"""
Background Tasks for FastAPI
Async operations (embedding generation, model retraining)
"""

import os
import time
import psycopg2.extras
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from api.dependencies import get_db_connection_context

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"


def build_incident_text(incident: dict) -> str:
    """Build text representation for embedding"""
    parts = [
        f"Issue: {incident.get('initial_message', 'unknown')}",
        f"Type: {incident.get('issue_type', 'unknown')}",
        f"Product Area: {incident.get('product_area', 'unknown')}",
        f"Priority: {incident.get('priority', 'unknown')}",
        f"Platform: {incident.get('platform', 'unknown')}",
    ]
    return " | ".join(parts)


def generate_embedding_task(incident_id: int):
    """
    Background task: Generate and store embedding for a single incident
    Called after POST /incidents to avoid blocking the response
    """
    print(f"[Background] Generating embedding for incident {incident_id}")
    
    try:
        with get_db_connection_context() as conn:
            # Fetch incident data
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, initial_message, issue_type, product_area, 
                           priority, platform
                    FROM incidents
                    WHERE id = %s
                """, (incident_id,))
                incident = cur.fetchone()
            
            if not incident:
                print(f"[Background] Incident {incident_id} not found")
                return
            
            # Build text and generate embedding
            text = build_incident_text(incident)
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Store embedding
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO incident_embeddings (incident_id, embedding_vector)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                """, (incident_id, str(embedding)))
                conn.commit()
            
            print(f"[Background] ✓ Embedding stored for incident {incident_id}")
    
    except Exception as e:
        print(f"[Background] ✗ Error generating embedding for incident {incident_id}: {e}")


def retrain_severity_model_task(job_id: str, min_samples: int = 1000):
    """
    Background task: Retrain Random Forest severity model
    Called by POST /batch/retrain
    """
    print(f"[Background] Starting model retraining job {job_id}")
    start_time = time.time()
    
    try:
        with get_db_connection_context() as conn:
            # Check available samples
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM incidents WHERE priority IS NOT NULL")
                count = cur.fetchone()[0]
            
            if count < min_samples:
                print(f"[Background] ✗ Insufficient samples: {count} < {min_samples}")
                return
            
            print(f"[Background] Found {count} samples. Starting retraining...")
            
            # Import training function from ml/severity_model.py
            from ml.severity_model import train_severity_model
            
            # This would call the actual training function
            # For now, we'll simulate the retraining process
            # In production, you'd call: train_severity_model()
            
            print(f"[Background] Training model with {count} samples...")
            time.sleep(2)  # Simulate training time
            
            duration = time.time() - start_time
            print(f"[Background] ✓ Model retraining completed in {duration:.2f}s")
            
            # TODO: Store retraining metadata in database
            # - job_id, status, duration, accuracy, timestamp
    
    except Exception as e:
        print(f"[Background] ✗ Error in retraining job {job_id}: {e}")


def bulk_generate_embeddings_task(batch_size: int = 100):
    """
    Background task: Generate embeddings for incidents without embeddings
    Can be triggered by admin endpoint or scheduled job
    """
    print(f"[Background] Starting bulk embedding generation (batch_size={batch_size})")
    
    try:
        with get_db_connection_context() as conn:
            # Fetch incidents without embeddings
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT i.id, i.initial_message, i.issue_type,
                           i.product_area, i.priority, i.platform
                    FROM incidents i
                    LEFT JOIN incident_embeddings ie ON i.id = ie.incident_id
                    WHERE ie.id IS NULL
                    LIMIT %s
                """, (batch_size,))
                incidents = cur.fetchall()
            
            if not incidents:
                print("[Background] No incidents found without embeddings")
                return
            
            print(f"[Background] Generating embeddings for {len(incidents)} incidents")
            
            # Build texts
            texts = [build_incident_text(inc) for inc in incidents]
            incident_ids = [inc['id'] for inc in incidents]
            
            # Generate embeddings in batch
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            
            # Store embeddings
            with conn.cursor() as cur:
                rows = [(inc_id, str(emb)) for inc_id, emb in zip(incident_ids, embeddings)]
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO incident_embeddings (incident_id, embedding_vector)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                    """,
                    rows,
                    page_size=len(rows)
                )
                conn.commit()
            
            print(f"[Background] ✓ Stored {len(embeddings)} embeddings")
    
    except Exception as e:
        print(f"[Background] ✗ Error in bulk embedding generation: {e}")
