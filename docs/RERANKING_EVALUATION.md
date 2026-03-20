# Reranking Evaluation

DecisionLens compares two retrieval pipelines on the same labeled query set:

- `baseline`: plain semantic search using raw pgvector similarity order
- `reranked`: semantic search followed by custom reranking with similarity, status, recency, and priority weighting

## Labeled query set

The checked-in query set lives in [benchmarks/labeled_queries.json](/Users/priya/Desktop/Projects/DecisionLens/benchmarks/labeled_queries.json).
Each query includes a list of known relevant ticket IDs drawn from repeated resolved incidents in the dataset.

## Run the evaluation

```bash
./venv/bin/python scripts/evaluate_reranking.py
```

This writes results to:

`benchmarks/reranking_eval_results.json`

## Metrics

- `hit_at_k`: whether any relevant ticket appears in the top results
- `mrr`: reciprocal rank of the first relevant result
- `precision_at_k`: proportion of relevant tickets in the returned top results

These metrics let us compare the plain semantic-search baseline against the custom reranker on the same labeled set.
