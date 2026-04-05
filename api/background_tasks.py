"""
Background Tasks for FastAPI
Async operations for embedding generation
"""

import os
import psycopg2.extras
from openai import OpenAI
from dotenv import load_dotenv
from api.dependencies import get_db_connection_context
from api.telemetry import emit_latency, now_ms, elapsed_ms

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"


def normalize_support_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = str(text).lower().strip()
    normalized = normalized.replace("can't", "cannot")
    normalized = normalized.replace("unable to", "cannot")
    normalized = normalized.replace("not able to", "cannot")
    normalized = normalized.replace("log in", "login")
    normalized = normalized.replace("sign in", "login")
    normalized = normalized.replace("password reset", "reset password")
    normalized = " ".join(normalized.split())
    return normalized


def build_incident_text(incident: dict) -> str:
    """Build text representation for embedding"""
    parts = [
        f"Issue: {normalize_support_text(incident.get('initial_message')) or 'unknown'}",
        f"Type: {normalize_support_text(incident.get('issue_type')) or 'unknown'}",
        f"Product Area: {normalize_support_text(incident.get('product_area')) or 'unknown'}",
        f"Priority: {incident.get('priority', 'unknown')}",
        f"Platform: {normalize_support_text(incident.get('platform')) or 'unknown'}",
    ]
    return " | ".join(parts)


def generate_embedding_task(incident_id: int):
    """
    Background task: Generate and store embedding for a single incident
    Called after POST /incidents to avoid blocking the response
    """
    print(f"[Background] Generating embedding for incident {incident_id}")
    task_start_ms = now_ms()
    
    try:
        with get_db_connection_context() as conn:
            # Fetch incident data
            fetch_start_ms = now_ms()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, initial_message, issue_type, product_area, 
                           priority, platform
                    FROM incidents
                    WHERE id = %s
                """, (incident_id,))
                incident = cur.fetchone()
            fetch_duration_ms = elapsed_ms(fetch_start_ms)
            
            if not incident:
                print(f"[Background] Incident {incident_id} not found")
                emit_latency(
                    "embedding_generation",
                    component="background_embedding_task",
                    incident_id=incident_id,
                    fetch_duration_ms=fetch_duration_ms,
                    embed_duration_ms=0,
                    store_duration_ms=0,
                    duration_ms=elapsed_ms(task_start_ms),
                    status="not_found",
                )
                return
            
            # Build text and generate embedding
            text = build_incident_text(incident)
            embed_start_ms = now_ms()
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            embedding = response.data[0].embedding
            embed_duration_ms = elapsed_ms(embed_start_ms)
            
            # Store embedding
            store_start_ms = now_ms()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO incident_embeddings (incident_id, embedding_vector)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                """, (incident_id, str(embedding)))
                conn.commit()
            store_duration_ms = elapsed_ms(store_start_ms)
            
            print(f"[Background] ✓ Embedding stored for incident {incident_id}")
            emit_latency(
                "embedding_generation",
                component="background_embedding_task",
                incident_id=incident_id,
                fetch_duration_ms=fetch_duration_ms,
                embed_duration_ms=embed_duration_ms,
                store_duration_ms=store_duration_ms,
                duration_ms=elapsed_ms(task_start_ms),
                status="success",
            )
    
    except Exception as e:
        print(f"[Background] ✗ Error generating embedding for incident {incident_id}: {e}")
        emit_latency(
            "embedding_generation",
            component="background_embedding_task",
            incident_id=incident_id,
            duration_ms=elapsed_ms(task_start_ms),
            status="error",
            error=str(e),
        )

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
