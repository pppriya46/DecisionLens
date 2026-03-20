import os
import psycopg2
import psycopg2.extras
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from api.telemetry import emit_latency, now_ms, elapsed_ms

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "decisionlens_db"),
    "user":     os.getenv("DB_USER", "decisionlens"),
    "password": os.getenv("DB_PASSWORD", "decisionlens123"),
}

EMBEDDING_MODEL = "text-embedding-3-small"
PRIORITY_WEIGHTS = {
    "critical": 1.0,
    "urgent": 0.9,
    "high": 0.75,
    "medium": 0.55,
    "low": 0.35,
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def generate_query_embedding(query_text: str):
    start_ms = now_ms()
    enriched_query = (
        f"User problem: {query_text}. "
        f"Looking for similar technical support issues, their solutions, and troubleshooting steps."
    )
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=enriched_query
    )
    emit_latency(
        "embedding_generation",
        component="search_query_embedding",
        model=EMBEDDING_MODEL,
        query_length=len(query_text),
        duration_ms=elapsed_ms(start_ms),
    )
    return response.data[0].embedding


def normalize_query_embedding(query_embedding):
    if isinstance(query_embedding, str):
        return query_embedding
    return str(query_embedding)


def score_priority(priority: str | None) -> float:
    normalized = (priority or "").strip().lower()
    return PRIORITY_WEIGHTS.get(normalized, 0.2)


def query_similar_incidents(conn, query_embedding, top_k=20):
    start_ms = now_ms()
    embedding_str = normalize_query_embedding(query_embedding)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                i.id,
                i.ticket_id,
                i.status,
                i.issue_type,
                i.product_area,
                i.priority,
                i.initial_message,
                i.resolution_summary,
                i.created_at,
                i.resolution_time_hours,
                i.customer_sentiment,
                i.csat_score,
                1 - (ie.embedding_vector <=> %s::vector) AS similarity_score
            FROM incident_embeddings ie
            JOIN incidents i ON ie.incident_id = i.id
            WHERE i.resolution_summary IS NOT NULL
            AND i.resolution_summary != ''
            ORDER BY ie.embedding_vector <=> %s::vector ASC
            LIMIT %s
        """, (embedding_str, embedding_str, top_k))

        results = cur.fetchall()

    emit_latency(
        "vector_search",
        component="pgvector_similarity_search",
        top_k=top_k,
        candidates_returned=len(results),
        duration_ms=elapsed_ms(start_ms),
    )
    return results


def rerank_incidents(incidents, query_category=None, top_n=5):
    start_ms = now_ms()
    now = datetime.now()
    scored = []

    for incident in incidents:
        base_score = float(incident['similarity_score'])

        state = (incident['status'] or '').lower()
        if state == 'resolved':
            status_score = 1.0
        elif state == 'in_progress':
            status_score = 0.7
        elif state in ('open', 'on_hold'):
            status_score = 0.5
        else:
            status_score = 0.3

        created_date = incident['created_at']
        if created_date:
            days_ago = (now - created_date).days
            recency_score = max(0.1, 1.0 - (days_ago / 365))
        else:
            recency_score = 0.1

        priority_score = score_priority(incident.get('priority'))

        final_score = (
            base_score     * 0.50 +
            status_score   * 0.20 +
            recency_score  * 0.10 +
            priority_score * 0.20
        )

        scored.append({
            "id":           incident['id'],
            "ticket_id":    incident['ticket_id'],
            "status":       incident['status'],
            "issue_type":   incident['issue_type'],
            "product_area": incident['product_area'],
            "priority":     incident['priority'],
            "description":  incident['initial_message'],
            "resolution":   incident['resolution_summary'],
            "created_at":   str(incident['created_at']) if incident['created_at'] else None,
            "sentiment":    incident['customer_sentiment'],
            "csat_score":   incident['csat_score'],
            "scores": {
                "final":      round(final_score, 4),
                "similarity": round(base_score, 4),
                "status":     round(status_score, 4),
                "recency":    round(recency_score, 4),
                "priority":   round(priority_score, 4),
            }
        })

    scored.sort(key=lambda x: x['scores']['final'], reverse=True)
    ranked = scored[:top_n]
    emit_latency(
        "rerank",
        component="incident_reranking",
        input_count=len(incidents),
        output_count=len(ranked),
        query_category=query_category,
        duration_ms=elapsed_ms(start_ms),
    )
    return ranked


def search_similar_incidents(
    query_text: str,
    query_category: str = None,
    query_embedding=None,
    top_k: int = 20,
    top_n: int = 5
) -> dict:
    print(f"\nSearch query: {query_text[:80]}...")
    request_start_ms = now_ms()

    conn = get_db_connection()

    try:
        embedding_start_ms = now_ms()
        if query_embedding is None:
            print("Generating enriched query embedding...")
            query_embedding = generate_query_embedding(query_text)
            embedding_duration_ms = elapsed_ms(embedding_start_ms)
            embedding_source = "generated"
        else:
            print("Reusing stored incident embedding...")
            embedding_duration_ms = elapsed_ms(embedding_start_ms)
            embedding_source = "stored"
            emit_latency(
                "embedding_generation",
                component="search_query_embedding",
                model="stored_incident_embedding",
                query_length=len(query_text),
                duration_ms=embedding_duration_ms,
                source=embedding_source,
            )

        print(f"Querying pgvector for top {top_k} candidates...")
        vector_search_start_ms = now_ms()
        raw_results = query_similar_incidents(conn, query_embedding, top_k)
        vector_search_duration_ms = elapsed_ms(vector_search_start_ms)
        print(f"Found {len(raw_results)} raw candidates")

        if not raw_results:
            total_duration_ms = elapsed_ms(request_start_ms)
            emit_latency(
                "search_request",
                component="search_similar_incidents",
                query_category=query_category,
                top_k=top_k,
                top_n=top_n,
                total_candidates=0,
                returned_results=0,
                timings_ms={
                    "embedding": embedding_duration_ms,
                    "vector_search": vector_search_duration_ms,
                    "rerank": 0,
                    "total": total_duration_ms,
                },
                embedding_source=embedding_source,
                duration_ms=total_duration_ms,
            )
            return {
                "query":            query_text,
                "total_candidates": 0,
                "results":          [],
                "message":          "No similar incidents found",
                "timings_ms": {
                    "embedding": embedding_duration_ms,
                    "vector_search": vector_search_duration_ms,
                    "rerank": 0,
                    "total": total_duration_ms,
                },
                "embedding_source": embedding_source,
            }

        print("Re-ranking by status and recency...")
        rerank_start_ms = now_ms()
        ranked_results = rerank_incidents(raw_results, query_category, top_n)
        rerank_duration_ms = elapsed_ms(rerank_start_ms)
        print(f"Returning top {len(ranked_results)} results")

        total_duration_ms = elapsed_ms(request_start_ms)
        timings_ms = {
            "embedding": embedding_duration_ms,
            "vector_search": vector_search_duration_ms,
            "rerank": rerank_duration_ms,
            "total": total_duration_ms,
        }
        emit_latency(
            "search_request",
            component="search_similar_incidents",
            query_category=query_category,
            top_k=top_k,
            top_n=top_n,
            total_candidates=len(raw_results),
            returned_results=len(ranked_results),
            timings_ms=timings_ms,
            embedding_source=embedding_source,
            duration_ms=total_duration_ms,
        )

        return {
            "query":          query_text,
            "query_category": query_category,
            "total_candidates": len(raw_results),
            "results":        ranked_results,
            "timings_ms":     timings_ms,
            "embedding_source": embedding_source,
        }

    finally:
        conn.close()


if __name__ == "__main__":
    print("Testing Similarity Search Engine...\n")

    result = search_similar_incidents(
        query_text="I cannot login to my account",
        query_category="account_access",
        top_k=20,
        top_n=5
    )

    print("\n" + "="*60)
    print("SEARCH RESULTS")
    print("="*60)
    print(f"Total pgvector candidates : {result['total_candidates']}")
    print(f"Returned after re-ranking : {len(result['results'])}\n")

    for i, inc in enumerate(result['results'], 1):
        print(f"{i}. {inc['ticket_id']} [{inc['status']}]")
        print(f"   Issue Type : {inc['issue_type']} | {inc['product_area']}")
        print(f"   Description: {inc['description'][:80]}...")
        print(f"   Priority   : {inc['priority']}")
        print(f"   Resolution : {inc['resolution'][:80] if inc['resolution'] else 'N/A'}...")
        print(f"   Scores     : {inc['scores']}")
        print()
