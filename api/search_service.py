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
    """Create and return a database connection."""
    return psycopg2.connect(**DB_CONFIG)


def generate_query_embedding(query_text: str):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query_text
    )
    return response.data[0].embedding


def query_similar_incidents(conn, query_embedding, top_k=20):
    """
    Query pgvector using HNSW index for fast nearest-neighbor search.
    Fetches top_k=20 so re-ranking has enough candidates to work with.
    Uses <-> operator = cosine distance (lower = more similar)
    """
    embedding_str = str(query_embedding)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                i.id,
                i.number,
                i.incident_state,
                i.category,
                i.subcategory,
                i.u_symptom,
                i.priority,
                i.impact,
                i.urgency,
                i.closed_code,
                i.opened_at,
                i.resolved_at,
                i.closed_at,
                1 - (ie.embedding_vector <-> %s::vector) AS similarity_score
            FROM incident_embeddings ie
            JOIN incidents i ON ie.incident_id = i.id
            ORDER BY ie.embedding_vector <-> %s::vector ASC
            LIMIT %s
        """, (embedding_str, embedding_str, top_k))

        return cur.fetchall()


# ── Step 3: Re-rank Results ─────────────────────────────────────────────
def rerank_incidents(incidents, query_category=None, top_n=5):
    """
    Re-rank raw pgvector results by combining 4 signals:

    1. Similarity score : base cosine similarity from pgvector  (weight: 50%)
    2. Status filter    : prefer resolved/closed incidents       (weight: 25%)
    3. Recency weight   : prefer more recently resolved          (weight: 15%)
    4. Category match   : boost same category as query          (weight: 10%)
    """
    now = datetime.now()
    scored = []

    for incident in incidents:
        base_score = float(incident['similarity_score'])

        # ── Signal 1: Status Filter ────────────────────────────────────
        # Prefer closed/resolved - they have actual solutions
        state = (incident['incident_state'] or '').lower()
        if state in ('closed', 'resolved'):
            status_score = 1.0    # has a resolution
        elif state == 'open':
            status_score = 0.3    # no resolution yet
        else:
            status_score = 0.5    # unknown

        # ── Signal 2: Recency Weight ───────────────────────────────────
        # Recently resolved = more relevant to current systems/processes
        resolved_date = incident['resolved_at'] or incident['closed_at']
        if resolved_date:
            days_ago = (now - resolved_date).days
            recency_score = max(0.1, 1.0 - (days_ago / 365))
        else:
            recency_score = 0.1   # no resolution date

        # ── Signal 3: Category Match ───────────────────────────────────
        # Boost incidents from same category as query
        if query_category:
            incident_cat = (incident['category'] or '').lower()
            query_cat    = query_category.lower()

            if incident_cat == query_cat:
                category_score = 1.0    # exact match
            elif query_cat in incident_cat:
                category_score = 0.7    # partial match
            else:
                category_score = 0.3    # no match
        else:
            category_score = 0.5        # no category provided

        # ── Final Score: Weighted Combination ──────────────────────────
        final_score = (
            base_score     * 0.50 +
            status_score   * 0.25 +
            recency_score  * 0.15 +
            category_score * 0.10
        )

        scored.append({
            "id":             incident['id'],
            "number":         incident['number'],
            "incident_state": incident['incident_state'],
            "category":       incident['category'],
            "subcategory":    incident['subcategory'],
            "symptom":        incident['u_symptom'],
            "priority":       incident['priority'],
            "impact":         incident['impact'],
            "urgency":        incident['urgency'],
            "resolution":     incident['closed_code'],
            "opened_at":      str(incident['opened_at'])   if incident['opened_at']   else None,
            "resolved_at":    str(incident['resolved_at']) if incident['resolved_at'] else None,
            "scores": {
                "final":      round(final_score, 4),
                "similarity": round(base_score, 4),
                "status":     round(status_score, 4),
                "recency":    round(recency_score, 4),
                "category":   round(category_score, 4),
            }
        })

    # Sort by final score, return top N
    scored.sort(key=lambda x: x['scores']['final'], reverse=True)
    return scored[:top_n]


# ── Main Search Function ────────────────────────────────────────────────
def search_similar_incidents(
    query_text: str,
    query_category: str = None,
    top_k: int = 20,
    top_n: int = 5
) -> dict:
    """
    Full similarity search pipeline:
    1. Generate query embedding
    2. Query pgvector (HNSW index) for top-K candidates
    3. Re-rank by status, recency, category
    4. Return top 5 most relevant incidents

    Args:
        query_text     : incident description to search for
        query_category : optional category hint for re-ranking
        top_k          : raw candidates to fetch from pgvector (default 20)
        top_n          : final results to return after re-ranking (default 5)
    """
    print(f"\nSearch query: {query_text[:80]}...")

    conn = get_db_connection()

    try:
        # Step 1: Embed the query
        print("Generating query embedding...")
        query_embedding = generate_query_embedding(query_text)

        # Step 2: Query pgvector
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

        # Step 3: Re-rank
        print("Re-ranking by status, recency, category...")
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


# ── Manual Test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Similarity Search Engine...\n")

    result = search_similar_incidents(
        query_text="User cannot login, password reset not working",
        query_category="Software",
        top_k=20,
        top_n=5
    )

    print("\n" + "="*60)
    print("SEARCH RESULTS")
    print("="*60)
    print(f"Total pgvector candidates : {result['total_candidates']}")
    print(f"Returned after re-ranking : {len(result['results'])}\n")

    for i, inc in enumerate(result['results'], 1):
        print(f"{i}. {inc['number']} [{inc['incident_state']}]")
        print(f"   Category   : {inc['category']} > {inc['subcategory']}")
        print(f"   Symptom    : {inc['symptom']}")
        print(f"   Priority   : {inc['priority']}")
        print(f"   Resolution : {inc['resolution']}")
        print(f"   Scores     : {inc['scores']}")
        print()

    print("="*60)

    