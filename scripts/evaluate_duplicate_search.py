#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import psycopg2.extras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.search_service import (
    build_result_group_key,
    get_db_connection,
    normalize_support_text,
    search_duplicate_incidents,
)


def load_eval_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fetch_ticket_records(ticket_ids):
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, ticket_id, initial_message, issue_type, product_area
                FROM incidents
                WHERE ticket_id = ANY(%s)
                """,
                (ticket_ids,),
            )
            rows = cur.fetchall()
            return {row["ticket_id"]: row for row in rows}
    finally:
        conn.close()


def build_ticket_signature(record):
    return build_result_group_key(
        record.get("initial_message"),
        record.get("issue_type"),
        record.get("product_area"),
    )


def build_eval_ticket_signature(row):
    return build_result_group_key(
        row.get("candidate_text"),
        None,
        None,
    )


def evaluate_pair(row, query_record, candidate_record, top_k, top_n):
    result = search_duplicate_incidents(
        query_text=query_record["initial_message"],
        query_issue_type=query_record.get("issue_type"),
        query_product_area=query_record.get("product_area"),
        top_k=top_k,
        top_n=top_n,
    )
    returned_ticket_ids = []
    returned_signatures = []
    for item in result["results"]:
        grouped_ticket_ids = item.get("grouped_ticket_ids") or [item["ticket_id"]]
        for ticket_id in grouped_ticket_ids:
            if ticket_id not in returned_ticket_ids:
                returned_ticket_ids.append(ticket_id)
        returned_signatures.append(
            build_result_group_key(
                item.get("description"),
                item.get("issue_type"),
                item.get("product_area"),
            )
        )
    found = row["candidate_ticket_id"] in returned_ticket_ids
    rank = returned_ticket_ids.index(row["candidate_ticket_id"]) + 1 if found else None
    candidate_signature = build_ticket_signature(candidate_record) if candidate_record else build_eval_ticket_signature(row)
    semantic_found = candidate_signature in returned_signatures
    semantic_rank = returned_signatures.index(candidate_signature) + 1 if semantic_found else None
    return {
        "query_ticket_id": row["query_ticket_id"],
        "candidate_ticket_id": row["candidate_ticket_id"],
        "expected_label": row["expected_label"],
        "found": found,
        "rank": rank,
        "semantic_found": semantic_found,
        "semantic_rank": semantic_rank,
        "classification": result["classification"],
        "top_match_score": result["top_match_score"],
        "returned_ticket_ids": returned_ticket_ids,
        "returned_signatures": returned_signatures,
        "rationale": row["rationale"],
    }


def summarize_results(results):
    total = len(results)
    found_count = sum(1 for item in results if item["found"])
    semantic_found_count = sum(1 for item in results if item["semantic_found"])
    by_label = defaultdict(list)
    for item in results:
        by_label[item["expected_label"]].append(item)

    label_summary = {}
    for label, items in sorted(by_label.items()):
        hits = sum(1 for item in items if item["found"])
        semantic_hits = sum(1 for item in items if item["semantic_found"])
        label_summary[label] = {
            "total": len(items),
            "hits": hits,
            "hit_rate": round(hits / len(items), 4) if items else 0.0,
            "avg_rank_when_found": round(
                sum(item["rank"] for item in items if item["rank"] is not None) / hits,
                2,
            ) if hits else None,
            "semantic_hits": semantic_hits,
            "semantic_hit_rate": round(semantic_hits / len(items), 4) if items else 0.0,
            "avg_semantic_rank_when_found": round(
                sum(item["semantic_rank"] for item in items if item["semantic_rank"] is not None) / semantic_hits,
                2,
            ) if semantic_hits else None,
        }

    classification_counts = Counter(item["classification"] for item in results)
    return {
        "total_pairs": total,
        "hits": found_count,
        "overall_hit_rate": round(found_count / total, 4) if total else 0.0,
        "semantic_hits": semantic_found_count,
        "overall_semantic_hit_rate": round(semantic_found_count / total, 4) if total else 0.0,
        "by_label": label_summary,
        "classification_counts": dict(classification_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate duplicate search against the curated duplicate eval set.")
    parser.add_argument(
        "--labels",
        default="data/evals/duplicate_eval_set.csv",
        help="Path to the duplicate evaluation CSV.",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/duplicate_eval_results.json",
        help="Path to write JSON results.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--show-misses", type=int, default=10)
    args = parser.parse_args()

    labels_path = Path(args.labels)
    output_path = Path(args.output)
    rows = load_eval_rows(labels_path)

    ticket_ids = sorted(
        {
            row["query_ticket_id"]
            for row in rows
        } | {
            row["candidate_ticket_id"]
            for row in rows
        }
    )
    ticket_records = fetch_ticket_records(ticket_ids)

    query_ticket_ids = sorted({row["query_ticket_id"] for row in rows})
    missing_queries = [ticket_id for ticket_id in query_ticket_ids if ticket_id not in ticket_records]
    if missing_queries:
        raise SystemExit(f"Missing query tickets in database: {', '.join(missing_queries[:10])}")

    evaluations = [
        evaluate_pair(
            row,
            ticket_records[row["query_ticket_id"]],
            ticket_records.get(row["candidate_ticket_id"]),
            top_k=args.top_k,
            top_n=args.top_n,
        )
        for row in rows
    ]
    summary = summarize_results(evaluations)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "labels_path": str(labels_path),
                "top_k": args.top_k,
                "top_n": args.top_n,
                "summary": summary,
                "per_pair": evaluations,
            },
            indent=2,
        )
    )

    print("Duplicate Search Evaluation")
    print("=" * 32)
    print(f"Pairs evaluated : {summary['total_pairs']}")
    print(f"Exact hits      : {summary['hits']}")
    print(f"Exact hit rate  : {summary['overall_hit_rate']:.2%}")
    print(f"Semantic hits   : {summary['semantic_hits']}")
    print(f"Semantic hit rate: {summary['overall_semantic_hit_rate']:.2%}")
    print("\nBy label:")
    for label, stats in summary["by_label"].items():
        avg_rank = stats["avg_rank_when_found"] if stats["avg_rank_when_found"] is not None else "n/a"
        avg_semantic_rank = (
            stats["avg_semantic_rank_when_found"]
            if stats["avg_semantic_rank_when_found"] is not None
            else "n/a"
        )
        print(
            f"- {label}: exact {stats['hits']}/{stats['total']} ({stats['hit_rate']:.2%}), "
            f"avg exact rank={avg_rank}; semantic {stats['semantic_hits']}/{stats['total']} "
            f"({stats['semantic_hit_rate']:.2%}), avg semantic rank={avg_semantic_rank}"
        )

    print("\nReturned classifications:")
    for label, count in sorted(summary["classification_counts"].items()):
        print(f"- {label}: {count}")

    misses = [item for item in evaluations if not item["found"]][: args.show_misses]
    if misses:
        print("\nSample misses:")
        for item in misses:
            print(
                f"- query={item['query_ticket_id']} candidate={item['candidate_ticket_id']} "
                f"expected={item['expected_label']} returned={item['returned_ticket_ids'][:3]}"
            )

    print(f"\nSaved JSON results to {output_path}")


if __name__ == "__main__":
    main()
