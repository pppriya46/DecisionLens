# DecisionLens

## What it does

DecisionLens is an incident resolution system that helps surface similar historical support cases
and generate troubleshooting guidance for new incidents. It combines semantic search over past
tickets with an LLM-assisted resolution flow so support teams can investigate issues faster and
more consistently.

## Why I built this

I built this project to explore how retrieval, reranking, and LLM-based generation can work
together in a practical support workflow. I also wanted to turn a simple semantic-search idea
into a more complete system with measurable performance, evaluation, and deployment setup.

## How it works

- API layer: FastAPI exposes endpoints for incident creation, retrieval, similarity search,
  resolution generation, health checks, and ML status reporting.
- Database: PostgreSQL stores incidents, embeddings, and model-related metadata, with pgvector
  IVFFlat indexing used for semantic retrieval.
- Processing logic: incident text is embedded with OpenAI, similar resolved incidents are fetched
  with pgvector cosine search, results are reranked using operational signals like priority,
  status, and recency, and GPT-4 generates troubleshooting guidance from the top matches.

## Architecture

```text
Incident data
    |
    v
PostgreSQL + pgvector
    |
    +--> incident records
    +--> stored embedding vectors
    |
    v
FastAPI API
    |
    +--> create incident
    +--> fetch incident
    +--> search similar incidents
    +--> generate resolution
    |
    v
Retrieval pipeline
    |
    +--> OpenAI embeddings
    +--> pgvector similarity search
    +--> priority/status/recency reranking
    |
    v
GPT-4 resolution generation
    |
    v
API response + stored resolution
```

## Run locally

### Option 1: Docker

Start the app stack:

```bash
docker compose up --build
```

Available services:

- Frontend: `http://localhost:3000`
- API: `http://localhost:5000`
- API docs: `http://localhost:5000/docs`

On a fresh Docker volume, PostgreSQL initializes the schema automatically from
[db/schema.sql](/Users/priya/Desktop/Projects/DecisionLens/db/schema.sql).

If you want to reset the stack:

```bash
docker compose down -v
docker compose up --build
```

If you want to load the incident dataset:

```bash
docker compose run --rm api python data/load_incidents.py
```

If you want to generate embeddings:

```bash
docker compose run --rm api python ml/embedding_service.py
```

### Option 2: Manual

Start PostgreSQL first, then run:

```bash
source venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 5000
```

For the frontend:

```bash
cd frontend
npm install
npm run dev
```

## What I Learned

- Instrumentation matters. Adding latency breakdowns made it much easier to identify the real
  bottlenecks instead of guessing.
- Retrieval quality is not just about raw vector similarity. Reranking with operational signals
  made the results more useful without materially hurting semantic relevance.
- End-to-end performance depends on the full pipeline. Once retrieval became fast, the main
  latency bottleneck shifted to prompt size and LLM response time.
- Clean deployment and reproducibility are part of the project quality, not just the model logic.

## Future improvements

- Scaling: add stronger bulk-ingestion and background job orchestration for larger datasets and
  higher request volume.
- Monitoring: add richer production-style observability for latency, error tracking, and model
  usage over time.
- Retrieval quality: expand the evaluation set and improve relevance labeling beyond the current
  labeled benchmark set.
- Resolution generation: test faster model options and selective caching for lower end-to-end
  latency.
