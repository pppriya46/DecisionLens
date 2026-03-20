#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.search_service import (
    generate_query_embedding,
    get_db_connection,
    query_similar_incidents,
    rerank_incidents,
    score_priority,
)


def normalize_text(value):
    return " ".join((value or "").strip().lower().split())


def cluster_key(text, issue_type, product_area):
    return (
        normalize_text(text),
        (issue_type or "").strip().lower(),
        (product_area or "").strip().lower(),
    )


def reciprocal_rank(ticket_ids, relevant_ticket_ids):
    for index, ticket_id in enumerate(ticket_ids, start=1):
        if ticket_id in relevant_ticket_ids:
            return 1.0 / index
    return 0.0


def hit_at_k(ticket_ids, relevant_ticket_ids):
    return 1.0 if any(ticket_id in relevant_ticket_ids for ticket_id in ticket_ids) else 0.0


def precision_at_k(ticket_ids, relevant_ticket_ids):
    if not ticket_ids:
        return 0.0
    hits = sum(1 for ticket_id in ticket_ids if ticket_id in relevant_ticket_ids)
    return hits / len(ticket_ids)


def cluster_reciprocal_rank(results, expected_cluster):
    for index, item in enumerate(results, start=1):
        item_cluster = cluster_key(
            item.get("description") or item.get("initial_message"),
            item.get("issue_type"),
            item.get("product_area"),
        )
        if item_cluster == expected_cluster:
            return 1.0 / index
    return 0.0


def cluster_hit_at_k(results, expected_cluster):
    return 1.0 if cluster_reciprocal_rank(results, expected_cluster) > 0 else 0.0


def cluster_precision_at_k(results, expected_cluster):
    if not results:
        return 0.0
    hits = 0
    for item in results:
        item_cluster = cluster_key(
            item.get("description") or item.get("initial_message"),
            item.get("issue_type"),
            item.get("product_area"),
        )
        if item_cluster == expected_cluster:
            hits += 1
    return hits / len(results)


def candidate_relevance(candidate, query_record):
    description = candidate.get("description") or candidate.get("initial_message")
    same_text = description == query_record["query"]
    same_issue = candidate["issue_type"] == query_record.get("expected_issue_type")
    same_product = candidate["product_area"] == query_record.get("expected_product_area")

    if same_text and same_issue and same_product:
        return 3
    if same_text and same_issue:
        return 2
    if same_issue:
        return 1
    return 0


def dcg(relevances):
    total = 0.0
    for index, relevance in enumerate(relevances, start=1):
        total += relevance / math.log2(index + 1)
    return total


def ndcg_at_k(results, query_record):
    relevances = [candidate_relevance(item, query_record) for item in results]
    ideal = sorted(relevances, reverse=True)
    ideal_score = dcg(ideal)
    if ideal_score == 0:
        return 0.0
    return dcg(relevances) / ideal_score


def average_relevance(results, query_record):
    if not results:
        return 0.0
    return sum(candidate_relevance(item, query_record) for item in results) / len(results)


def score_status(status):
    normalized = (status or "").lower()
    if normalized == "resolved":
        return 1.0
    if normalized == "in_progress":
        return 0.7
    if normalized in ("open", "on_hold"):
        return 0.5
    return 0.3


def score_recency(created_at):
    if not created_at:
        return 0.1
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at.replace(" ", "T"))
    days_ago = (datetime.now() - created_at).days
    return max(0.1, 1.0 - (days_ago / 365))


def operational_metrics(results):
    if not results:
        return {
            "avg_similarity": 0.0,
            "avg_status_score": 0.0,
            "avg_priority_score": 0.0,
            "avg_recency_score": 0.0,
        }

    return {
        "avg_similarity": round(
            sum(float(item["similarity_score"] if "similarity_score" in item else item["scores"]["similarity"]) for item in results) / len(results),
            4,
        ),
        "avg_status_score": round(
            sum(score_status(item.get("status")) for item in results) / len(results),
            4,
        ),
        "avg_priority_score": round(
            sum(score_priority(item.get("priority")) for item in results) / len(results),
            4,
        ),
        "avg_recency_score": round(
            sum(score_recency(item.get("created_at")) for item in results) / len(results),
            4,
        ),
    }


def evaluate_query(query_record, top_k, top_n):
    query_text = query_record["query"]
    query_category = query_record.get("query_category")
    relevant_ticket_ids = set(query_record["relevant_ticket_ids"])
    expected_cluster = cluster_key(
        query_text,
        query_record.get("expected_issue_type"),
        query_record.get("expected_product_area"),
    )

    embedding = generate_query_embedding(query_text)
    conn = get_db_connection()
    try:
        raw_results = query_similar_incidents(conn, embedding, top_k=top_k)
    finally:
        conn.close()

    baseline_results = raw_results[:top_n]
    reranked_results = rerank_incidents(raw_results, query_category=query_category, top_n=top_n)

    baseline_ticket_ids = [item["ticket_id"] for item in baseline_results]
    reranked_ticket_ids = [item["ticket_id"] for item in reranked_results]

    return {
        "query": query_text,
        "query_category": query_category,
        "expected_issue_type": query_record.get("expected_issue_type"),
        "expected_product_area": query_record.get("expected_product_area"),
        "expected_cluster": list(expected_cluster),
        "relevant_ticket_ids": list(relevant_ticket_ids),
        "baseline": {
            "top_ticket_ids": baseline_ticket_ids,
            "hit_at_k": round(hit_at_k(baseline_ticket_ids, relevant_ticket_ids), 4),
            "mrr": round(reciprocal_rank(baseline_ticket_ids, relevant_ticket_ids), 4),
            "precision_at_k": round(precision_at_k(baseline_ticket_ids, relevant_ticket_ids), 4),
            "cluster_hit_at_k": round(cluster_hit_at_k(baseline_results, expected_cluster), 4),
            "cluster_mrr": round(cluster_reciprocal_rank(baseline_results, expected_cluster), 4),
            "cluster_precision_at_k": round(cluster_precision_at_k(baseline_results, expected_cluster), 4),
            "avg_relevance": round(average_relevance(baseline_results, query_record), 4),
            "ndcg_at_k": round(ndcg_at_k(baseline_results, query_record), 4),
            **operational_metrics(baseline_results),
        },
        "reranked": {
            "top_ticket_ids": reranked_ticket_ids,
            "hit_at_k": round(hit_at_k(reranked_ticket_ids, relevant_ticket_ids), 4),
            "mrr": round(reciprocal_rank(reranked_ticket_ids, relevant_ticket_ids), 4),
            "precision_at_k": round(precision_at_k(reranked_ticket_ids, relevant_ticket_ids), 4),
            "cluster_hit_at_k": round(cluster_hit_at_k(reranked_results, expected_cluster), 4),
            "cluster_mrr": round(cluster_reciprocal_rank(reranked_results, expected_cluster), 4),
            "cluster_precision_at_k": round(cluster_precision_at_k(reranked_results, expected_cluster), 4),
            "avg_relevance": round(average_relevance(reranked_results, query_record), 4),
            "ndcg_at_k": round(ndcg_at_k(reranked_results, query_record), 4),
            **operational_metrics(reranked_results),
        },
    }


def average_metric(results, pipeline_name, metric_name):
    if not results:
        return 0.0
    return round(
        sum(item[pipeline_name][metric_name] for item in results) / len(results),
        4,
    )


def main():
    parser = argparse.ArgumentParser(description="Compare baseline semantic search against custom reranking.")
    parser.add_argument(
        "--labels",
        default="benchmarks/labeled_queries.json",
        help="Path to the labeled query set.",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/reranking_eval_results.json",
        help="Path to write evaluation results.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    labels_path = Path(args.labels)
    output_path = Path(args.output)
    query_set = json.loads(labels_path.read_text())

    per_query_results = [
        evaluate_query(record, top_k=args.top_k, top_n=args.top_n)
        for record in query_set
    ]

    summary = {
        "query_count": len(per_query_results),
        "top_k": args.top_k,
        "top_n": args.top_n,
        "baseline": {
            "avg_hit_at_k": average_metric(per_query_results, "baseline", "hit_at_k"),
            "avg_mrr": average_metric(per_query_results, "baseline", "mrr"),
            "avg_precision_at_k": average_metric(per_query_results, "baseline", "precision_at_k"),
            "avg_cluster_hit_at_k": average_metric(per_query_results, "baseline", "cluster_hit_at_k"),
            "avg_cluster_mrr": average_metric(per_query_results, "baseline", "cluster_mrr"),
            "avg_cluster_precision_at_k": average_metric(per_query_results, "baseline", "cluster_precision_at_k"),
            "avg_relevance": average_metric(per_query_results, "baseline", "avg_relevance"),
            "avg_ndcg_at_k": average_metric(per_query_results, "baseline", "ndcg_at_k"),
            "avg_similarity": average_metric(per_query_results, "baseline", "avg_similarity"),
            "avg_status_score": average_metric(per_query_results, "baseline", "avg_status_score"),
            "avg_priority_score": average_metric(per_query_results, "baseline", "avg_priority_score"),
            "avg_recency_score": average_metric(per_query_results, "baseline", "avg_recency_score"),
        },
        "reranked": {
            "avg_hit_at_k": average_metric(per_query_results, "reranked", "hit_at_k"),
            "avg_mrr": average_metric(per_query_results, "reranked", "mrr"),
            "avg_precision_at_k": average_metric(per_query_results, "reranked", "precision_at_k"),
            "avg_cluster_hit_at_k": average_metric(per_query_results, "reranked", "cluster_hit_at_k"),
            "avg_cluster_mrr": average_metric(per_query_results, "reranked", "cluster_mrr"),
            "avg_cluster_precision_at_k": average_metric(per_query_results, "reranked", "cluster_precision_at_k"),
            "avg_relevance": average_metric(per_query_results, "reranked", "avg_relevance"),
            "avg_ndcg_at_k": average_metric(per_query_results, "reranked", "ndcg_at_k"),
            "avg_similarity": average_metric(per_query_results, "reranked", "avg_similarity"),
            "avg_status_score": average_metric(per_query_results, "reranked", "avg_status_score"),
            "avg_priority_score": average_metric(per_query_results, "reranked", "avg_priority_score"),
            "avg_recency_score": average_metric(per_query_results, "reranked", "avg_recency_score"),
        },
    }

    payload = {
        "summary": summary,
        "results": per_query_results,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
