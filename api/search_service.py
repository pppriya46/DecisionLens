import os
import re
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
LIKELY_DUPLICATE_THRESHOLD = 0.88
POSSIBLY_RELATED_THRESHOLD = 0.75
MIN_DISPLAY_DUPLICATE_SCORE = 0.6
MIN_DISPLAY_SIMILARITY_SCORE = 0.45
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "how", "i", "in", "is", "it", "my", "not", "of", "on",
    "or", "our", "so", "that", "the", "their", "them", "this", "to", "up",
    "we", "with", "you", "your",
}
TEXT_NORMALIZATION_RULES = [
    (r"\b(can ?not|can't|unable to|not able to)\b", "cannot"),
    (r"\b(log ?in|sign ?in)\b", "login"),
    (r"\bpassword reset\b", "reset password"),
    (r"\b(two times|twice)\b", "duplicate"),
    (r"\bkeeps? failing\b", "fails"),
    (r"\bnot working\b", "fails"),
]

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def normalize_support_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = text.lower().strip()
    for pattern, replacement in TEXT_NORMALIZATION_RULES:
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def generate_query_embedding(query_text: str, query_category: str | None = None):
    start_ms = now_ms()
    normalized_query = normalize_support_text(query_text)
    normalized_category = normalize_support_text(query_category)
    enriched_query = (
        f"User problem: {normalized_query}. "
        f"Issue category: {normalized_category or 'unknown'}. "
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
        has_query_category=bool(query_category),
        duration_ms=elapsed_ms(start_ms),
    )
    return response.data[0].embedding


def generate_duplicate_query_embedding(
    query_text: str,
    issue_type: str | None = None,
    product_area: str | None = None,
):
    start_ms = now_ms()
    normalized_query = normalize_support_text(query_text)
    normalized_issue_type = normalize_support_text(issue_type)
    normalized_product_area = normalize_support_text(product_area)
    enriched_query = " | ".join([
        f"Issue: {normalized_query or 'unknown'}",
        f"Type: {normalized_issue_type or 'unknown'}",
        f"Product Area: {normalized_product_area or 'unknown'}",
        "Priority: unknown",
        "Resolution: unknown",
        "Platform: unknown",
    ])

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=enriched_query,
    )
    emit_latency(
        "embedding_generation",
        component="duplicate_query_embedding",
        model=EMBEDDING_MODEL,
        query_length=len(query_text),
        has_issue_type=bool(issue_type),
        has_product_area=bool(product_area),
        duration_ms=elapsed_ms(start_ms),
    )
    return response.data[0].embedding


def normalize_query_embedding(query_embedding):
    if isinstance(query_embedding, str):
        return query_embedding
    return str(query_embedding)


def tokenize_for_overlap(text: str | None) -> set[str]:
    if not text:
        return set()
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_support_text(text))
        if len(token) > 2 and token not in STOPWORDS
    }
    return tokens


def score_keyword_overlap(query_text: str | None, candidate_text: str | None) -> float:
    query_tokens = tokenize_for_overlap(query_text)
    candidate_tokens = tokenize_for_overlap(candidate_text)

    if not query_tokens or not candidate_tokens:
        return 0.0

    overlap = len(query_tokens & candidate_tokens)
    return overlap / len(query_tokens)


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


def query_duplicate_candidates(conn, query_embedding, top_k=20):
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
                i.created_at,
                1 - (ie.embedding_vector <=> %s::vector) AS similarity_score
            FROM incident_embeddings ie
            JOIN incidents i ON ie.incident_id = i.id
            WHERE i.initial_message IS NOT NULL
            AND i.initial_message != ''
            ORDER BY ie.embedding_vector <=> %s::vector ASC
            LIMIT %s
        """, (embedding_str, embedding_str, top_k))

        results = cur.fetchall()

    emit_latency(
        "vector_search",
        component="pgvector_duplicate_search",
        top_k=top_k,
        candidates_returned=len(results),
        duration_ms=elapsed_ms(start_ms),
    )
    return results


def build_result_group_key(description: str | None, issue_type: str | None, product_area: str | None) -> str:
    normalized_description = normalize_support_text(description)
    normalized_issue_type = normalize_support_text(issue_type) or "unknown"
    normalized_product_area = normalize_support_text(product_area) or "unknown"
    tokens = tokenize_for_overlap(normalized_description)
    compact_description = " ".join(sorted(tokens)) if tokens else normalized_description
    return f"{normalized_issue_type}|{normalized_product_area}|{compact_description}"


def collapse_repetitive_results(results, *, top_n: int):
    grouped = {}

    for result in results:
        group_key = build_result_group_key(
            result.get("description"),
            result.get("issue_type"),
            result.get("product_area"),
        )

        if group_key not in grouped:
            grouped[group_key] = {
                **result,
                "related_ticket_count": 1,
            }
            continue

        grouped[group_key]["related_ticket_count"] += 1

    collapsed = list(grouped.values())
    collapsed.sort(
        key=lambda item: (
            item.get("scores", {}).get("final", item.get("duplicate_score", 0)),
            item.get("scores", {}).get("similarity", item.get("similarity_score", 0)),
        ),
        reverse=True,
    )
    return collapsed[:top_n]


def rerank_incidents(incidents, query_text: str | None = None, query_category=None, top_n=5):
    start_ms = now_ms()
    now = datetime.now()
    normalized_query_category = normalize_support_text(query_category)
    scored = []

    for incident in incidents:
        base_score = float(incident['similarity_score'])
        keyword_overlap = score_keyword_overlap(query_text, incident.get('initial_message'))
        category_match = 1.0 if (
            normalized_query_category and
            normalize_support_text(incident.get('issue_type')) == normalized_query_category
        ) else 0.0

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

        final_score = (
            base_score      * 0.55 +
            keyword_overlap * 0.15 +
            category_match  * 0.10 +
            status_score    * 0.15 +
            recency_score   * 0.05
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
            "related_ticket_count": 1,
            "scores": {
                "final":      round(final_score, 4),
                "similarity": round(base_score, 4),
                "keyword_overlap": round(keyword_overlap, 4),
                "category_match": round(category_match, 4),
                "status":     round(status_score, 4),
                "recency":    round(recency_score, 4),
            }
        })

    scored.sort(key=lambda x: x['scores']['final'], reverse=True)
    ranked = collapse_repetitive_results(scored, top_n=top_n)
    emit_latency(
        "rerank",
        component="incident_reranking",
        input_count=len(incidents),
        output_count=len(ranked),
        query_category=query_category,
        duration_ms=elapsed_ms(start_ms),
    )
    return ranked


def build_match_reasons(
    base_score: float,
    keyword_overlap: float,
    issue_match: float,
    product_match: float,
) -> list[str]:
    reasons = []

    if base_score >= 0.85:
        reasons.append("Very similar wording")
    elif base_score >= 0.75:
        reasons.append("Similar wording")
    elif base_score >= 0.6:
        reasons.append("Related wording")

    if keyword_overlap >= 0.45:
        reasons.append("Shared key terms")
    elif keyword_overlap >= 0.25:
        reasons.append("Some shared key terms")

    if issue_match:
        reasons.append("Same issue category")

    if product_match:
        reasons.append("Same affected area")

    if not reasons:
        reasons.append("Closest available historical match")

    return reasons


def rerank_duplicate_candidates(
    incidents,
    query_text: str | None = None,
    query_issue_type=None,
    query_product_area=None,
    top_n=5,
):
    start_ms = now_ms()
    query_issue_type = (query_issue_type or "").strip().lower()
    query_product_area = (query_product_area or "").strip().lower()
    scored = []

    for incident in incidents:
        base_score = float(incident["similarity_score"])
        issue_match = 1.0 if query_issue_type and (incident.get("issue_type") or "").strip().lower() == query_issue_type else 0.0
        product_match = 1.0 if query_product_area and (incident.get("product_area") or "").strip().lower() == query_product_area else 0.0
        keyword_overlap = score_keyword_overlap(query_text, incident.get("initial_message"))

        duplicate_score = (
            base_score * 0.72 +
            keyword_overlap * 0.18 +
            issue_match * 0.06 +
            product_match * 0.04
        )

        scored.append({
            "id": incident["id"],
            "ticket_id": incident["ticket_id"],
            "status": incident.get("status"),
            "issue_type": incident.get("issue_type"),
            "product_area": incident.get("product_area"),
            "priority": incident.get("priority"),
            "description": incident["initial_message"],
            "created_at": str(incident["created_at"]) if incident.get("created_at") else None,
            "similarity_score": round(base_score, 4),
            "keyword_overlap": round(keyword_overlap, 4),
            "duplicate_score": round(min(duplicate_score, 1.0), 4),
            "related_ticket_count": 1,
            "match_reasons": build_match_reasons(base_score, keyword_overlap, issue_match, product_match),
        })

    scored.sort(key=lambda x: x["duplicate_score"], reverse=True)
    ranked = collapse_repetitive_results(scored, top_n=top_n)
    emit_latency(
        "rerank",
        component="duplicate_candidate_reranking",
        input_count=len(incidents),
        output_count=len(ranked),
        query_issue_type=query_issue_type or None,
        query_product_area=query_product_area or None,
        duration_ms=elapsed_ms(start_ms),
    )
    return ranked


def filter_duplicate_candidates(ranked_results, classification):
    filtered = []

    for incident in ranked_results:
        base_score = incident.get("similarity_score", 0.0)
        duplicate_score = incident.get("duplicate_score", 0.0)
        keyword_overlap = incident.get("keyword_overlap", 0.0)

        if classification == "likely_duplicate":
            keep = duplicate_score >= POSSIBLY_RELATED_THRESHOLD
        elif classification == "possibly_related":
            keep = duplicate_score >= MIN_DISPLAY_DUPLICATE_SCORE
        else:
            keep = (
                duplicate_score >= MIN_DISPLAY_DUPLICATE_SCORE and
                (base_score >= MIN_DISPLAY_SIMILARITY_SCORE or keyword_overlap >= 0.25)
            )

        if keep:
            cleaned_incident = {
                key: value
                for key, value in incident.items()
                if key != "keyword_overlap"
            }
            if cleaned_incident.get("related_ticket_count", 1) > 1:
                cleaned_incident["match_reasons"] = [
                    *cleaned_incident.get("match_reasons", []),
                    f"{cleaned_incident['related_ticket_count']} similar tickets",
                ]
            filtered.append(cleaned_incident)

    return filtered


def classify_duplicate_match(top_match_score: float | None) -> tuple[str, str]:
    if top_match_score is None or top_match_score < POSSIBLY_RELATED_THRESHOLD:
        return (
            "new_issue",
            "No candidate exceeded the related-incident threshold.",
        )

    if top_match_score >= LIKELY_DUPLICATE_THRESHOLD:
        return (
            "likely_duplicate",
            "Top match score exceeded the duplicate threshold.",
        )

    return (
        "possibly_related",
        "Top match score exceeded the related-incident threshold but not the duplicate threshold.",
    )


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
            query_embedding = generate_query_embedding(query_text, query_category=query_category)
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
        ranked_results = rerank_incidents(raw_results, query_text=query_text, query_category=query_category, top_n=top_n)
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


def search_duplicate_incidents(
    query_text: str,
    query_issue_type: str = None,
    query_product_area: str = None,
    top_k: int = 20,
    top_n: int = 5,
) -> dict:
    print(f"\nDuplicate check query: {query_text[:80]}...")
    request_start_ms = now_ms()
    conn = get_db_connection()

    try:
        embedding_start_ms = now_ms()
        query_embedding = generate_duplicate_query_embedding(
            query_text,
            issue_type=query_issue_type,
            product_area=query_product_area,
        )
        embedding_duration_ms = elapsed_ms(embedding_start_ms)

        print(f"Querying pgvector for top {top_k} duplicate candidates...")
        vector_search_start_ms = now_ms()
        raw_results = query_duplicate_candidates(conn, query_embedding, top_k)
        vector_search_duration_ms = elapsed_ms(vector_search_start_ms)
        print(f"Found {len(raw_results)} raw duplicate candidates")

        rerank_start_ms = now_ms()
        ranked_results = rerank_duplicate_candidates(
            raw_results,
            query_text=query_text,
            query_issue_type=query_issue_type,
            query_product_area=query_product_area,
            top_n=top_n,
        )
        rerank_duration_ms = elapsed_ms(rerank_start_ms)

        top_match_score = ranked_results[0]["duplicate_score"] if ranked_results else None
        classification, explanation = classify_duplicate_match(top_match_score)
        filtered_results = filter_duplicate_candidates(ranked_results, classification)

        if not filtered_results and classification != "likely_duplicate":
            explanation = "No close historical match met the review threshold."

        total_duration_ms = elapsed_ms(request_start_ms)
        timings_ms = {
            "embedding": embedding_duration_ms,
            "vector_search": vector_search_duration_ms,
            "rerank": rerank_duration_ms,
            "total": total_duration_ms,
        }

        emit_latency(
            "duplicate_search_request",
            component="search_duplicate_incidents",
            query_issue_type=query_issue_type,
            query_product_area=query_product_area,
            top_k=top_k,
            top_n=top_n,
            total_candidates=len(raw_results),
            returned_results=len(filtered_results),
            classification=classification,
            timings_ms=timings_ms,
            duration_ms=total_duration_ms,
        )

        return {
            "query": query_text,
            "classification": classification,
            "explanation": explanation,
            "top_match_score": round(top_match_score, 4) if top_match_score is not None else 0.0,
            "total_candidates": len(raw_results),
            "results": filtered_results,
            "timings_ms": timings_ms,
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
