import os
import psycopg2
import psycopg2.extras
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

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

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def generate_query_embedding(query_text: str):
    """
    Enrich query with context for better similarity matching
    """
    # Expand query to match how incidents are embedded
    enriched_query = (
        f"User problem: {query_text}. "
        f"Looking for similar technical support issues, their solutions, and troubleshooting steps."
    )
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=enriched_query
    )
    return response.data[0].embedding


def query_similar_incidents(conn, query_embedding, top_k=20):
    embedding_str = str(query_embedding)

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
            ORDER BY ie.embedding_vector <=> %s::vector ASC
            LIMIT %s
        """, (embedding_str, embedding_str, top_k))

        return cur.fetchall()


def rerank_incidents(incidents, query_category=None, top_n=5):
    now = datetime.now()
    scored = []

    for incident in incidents:
        base_score = float(incident['similarity_score'])

        # Status score
        state = (incident['status'] or '').lower()
        if state == 'resolved':
            status_score = 1.0
        elif state == 'in_progress':
            status_score = 0.7
        elif state in ('open', 'on_hold'):
            status_score = 0.5
        else:
            status_score = 0.3

        # Recency score
        created_date = incident['created_at']
        if created_date:
            days_ago = (now - created_date).days
            recency_score = max(0.1, 1.0 - (days_ago / 365))
        else:
            recency_score = 0.1

        # Final score
        final_score = (
            base_score   * 0.60 +
            status_score * 0.25 +
            recency_score * 0.15
        )

        scored.append({
            "id":              incident['id'],
            "ticket_id":       incident['ticket_id'],
            "status":          incident['status'],
            "issue_type":      incident['issue_type'],
            "product_area":    incident['product_area'],
            "priority":        incident['priority'],
            "description":     incident['initial_message'],
            "resolution":      incident['resolution_summary'],
            "created_at":      str(incident['created_at']) if incident['created_at'] else None,
            "sentiment":       incident['customer_sentiment'],
            "csat_score":      incident['csat_score'],
            "scores": {
                "final":       round(final_score, 4),
                "similarity":  round(base_score, 4),
                "status":      round(status_score, 4),
                "recency":     round(recency_score, 4),
            }
        })

    scored.sort(key=lambda x: x['scores']['final'], reverse=True)
    return scored[:top_n]


def search_similar_incidents(
    query_text: str,
    query_category: str = None,
    top_k: int = 20,
    top_n: int = 5
) -> dict:
    print(f"\nSearch query: {query_text[:80]}...")

    conn = get_db_connection()

    try:
        print("Generating enriched query embedding...")
        query_embedding = generate_query_embedding(query_text)

        print(f"Querying pgvector for top {top_k} candidates...")
        raw_results = query_similar_incidents(conn, query_embedding, top_k)
        print(f"Found {len(raw_results)} raw candidates")

        if not raw_results:
            return {
                "query":            query_text,
                "total_candidates": 0,
                "results":          [],
                "message":          "No similar incidents found"
            }

        print("Re-ranking by status and recency...")
        ranked_results = rerank_incidents(raw_results, query_category, top_n)
        print(f"Returning top {len(ranked_results)} results")

        return {
            "query":            query_text,
            "query_category":   query_category,
            "total_candidates": len(raw_results),
            "results":          ranked_results,
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