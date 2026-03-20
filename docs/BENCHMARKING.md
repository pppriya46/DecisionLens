# Benchmarking

DecisionLens now includes endpoint-level latency instrumentation and a simple client-side
benchmark script for measuring response time distributions.

## What benchmarking means here

For this project, benchmarking means sending the same API request multiple times and recording
how long each response takes from the client perspective. We summarize those measurements with:

- `median (p50)`: the middle response time
- `p95`: the response time that 95% of requests are faster than
- `p99`: the slower tail of the latency distribution

This matters because one fast request does not prove the system is consistently fast. A resume
claim like `sub-5s P95 latency` requires repeated measurements, not a single best-case run.

## What optimization means here

Optimization means using the benchmark results and server-side timing logs to find the slowest
parts of each request path, then reducing their cost.

For DecisionLens, the main latency buckets are:

- query embedding generation
- pgvector similarity search
- reranking
- LLM response generation

The current instrumentation already emits structured JSON logs for those stages.

## How to run the benchmark

Start PostgreSQL and the API first, then run:

```bash
./venv/bin/python scripts/benchmark_api.py --endpoint all --runs 5 --incident-id 1
```

Benchmark only semantic search:

```bash
./venv/bin/python scripts/benchmark_api.py --endpoint similar --runs 10 --incident-id 1
```

Benchmark only resolution generation:

```bash
./venv/bin/python scripts/benchmark_api.py --endpoint resolve --runs 5 --incident-id 1
```

The script prints JSON with request counts and latency summaries, including `median`, `p95`,
and `p99`.

## How to read the results

Use the benchmark summary and the server logs together:

- If `/similar` is slow, inspect `embedding_generation` versus `vector_search`.
- If `/resolve` is slow, compare `retrieval_duration_ms` and `llm_duration_ms`.
- If `vector_search` is low but total latency is high, the bottleneck is not PostgreSQL.
- If `llm_duration_ms` dominates, search optimization alone will not deliver sub-5s end-to-end latency.

## Likely optimization opportunities in this codebase

- Reuse the stored incident embedding for `/incidents/{id}/similar` and `/incidents/{id}/resolve` instead of generating a fresh query embedding for an incident that already has one.
- Reduce prompt size for RAG by trimming source context.
- Evaluate a faster response-generation model if output quality remains acceptable.
- Cache repeated retrieval results for unchanged incidents.
- Keep pgvector index tuning (`lists`) under benchmark control rather than changing it blindly.
