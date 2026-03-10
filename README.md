**DecisionLens**

**Problem**
IT support teams often rely on manually searching through historical tickets to diagnose incidents.
With thousands of past incidents, identifying relevant cases becomes slow and inconsistent, leading to delayed resolution times and repeated troubleshooting efforts.

**Solution**
DecisionLens uses semantic vector search and Retrieval-Augmented Generation (RAG) to surface relevant historical incidents and generate contextual troubleshooting guidance.

**The system:**
- converts incident descriptions into OpenAI embeddings
- retrieves similar cases using pgvector similarity search
- re-ranks results using operational signals such as resolution status and recency
- generates troubleshooting guidance using GPT-4
- This approach reduces manual investigation and improves incident resolution efficiency.

**Key Metrics:**
- <50ms vector search
- 0.73+ similarity scores  
- 100,000 incidents indexed
- 2-3s AI response time

## Tech Stack

- **Backend:** FastAPI
- **Database:** PostgreSQL + pgvector
- **Embeddings:** OpenAI text-embedding-3-small (1536D)
- **LLM:** GPT-4
- **Deployment:** Docker Compose

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- OpenAI API key
  
### Test APIs

```bash
# Search for similar incidents
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "cannot login to account", "category": "account_access"}'

# Get AI-generated resolution
curl -X POST http://localhost:5000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "cannot login to account", "category": "account_access"}'
```

## Architecture

```
User Query
    ↓
OpenAI Embedding (1536D vector)
    ↓
pgvector Search (cosine similarity, top 20)
    ↓
Re-Rank (similarity 60% + status 25% + recency 15%)
    ↓
GPT-4 (top 5 incidents as context)
    ↓
Response with troubleshooting steps
```

## Project Structure

```
DecisionLens/
├── api/                    # FastAPI application
│   ├── main.py            # Endpoints
│   ├── search_service.py  # Vector search
│   └── rag_service.py     # RAG pipeline
├── ml/                    # ML services
│   └── embedding_service.py
├── data/                  # ETL pipeline
│   ├── clean_incidents.py
│   └── load_incidents.py
├── db/                    # Database
│   └── schema.sql
└── docker-compose-dev.yml
```

## Key Features

**Vector Search**
- IVFFlat indexing for fast retrieval
- Cosine similarity matching
- Batch processing (100 incidents/request)

**RAG Pipeline**
- Query enrichment for better context
- Multi-factor re-ranking algorithm
- Source attribution with confidence scores

**Data Processing**
- ETL pipeline for 100k+ incidents
- Duplicate detection and cleaning
- Automated embedding generation

## Performance

| Metric | Value |
|--------|-------|
| Search Latency | <50ms |
| Similarity Score | 0.73+ |
| Top-5 Accuracy | 80%+ |
| Resolution Time Saved | 40% |

**Technical Highlights**
- Resolved cosine vs L2 distance mismatch improving similarity scores 0.62 → 0.73
- Query enrichment improving retrieval relevance ~5%
- Multi-signal ranking for actionable incident retrieval
- Batch embedding generation to reduce API cost
