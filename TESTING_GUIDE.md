# DecisionLens Phase 2: FastAPI Migration - Testing Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server (Local)
```bash
# Using uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 5000

# Or using Docker Compose
docker-compose -f docker-compose-dev.yml up --build
```

### 3. Access Documentation
- **Swagger UI**: http://localhost:5000/docs
- **ReDoc**: http://localhost:5000/redoc
- **Health Check**: http://localhost:5000/health

---

## API Endpoints Testing

### **ENDPOINT 1: Create Incident** (POST /incidents)

**Description**: Create a new incident and trigger async embedding generation

```bash
curl -X POST "http://localhost:5000/incidents" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TKT-2026-0010",
    "initial_message": "Unable to connect to VPN from home network. Getting timeout error after authentication.",
    "customer_id": "CUST-9876",
    "channel": "email",
    "product_area": "Network Services",
    "issue_type": "VPN Access",
    "priority": "high",
    "platform": "Windows 11",
    "region": "US-West"
  }'
```

**Expected Response** (201 Created):
```json
{
  "incident_id": 100001,
  "ticket_id": "TKT-2026-0010",
  "status": "open",
  "message": "Incident created successfully. Embedding generation in progress.",
  "created_at": "2026-03-07T14:30:00"
}
```

**Test Cases**:
- ✅ Valid incident creation
- ❌ Duplicate ticket_id (409 Conflict)
- ❌ Missing required fields (422 Validation Error)

---

### **ENDPOINT 2: Get Incident Details** (GET /incidents/{id})

**Description**: Get full incident details with ML predictions and similar incidents

```bash
# Get incident with predictions and similar incidents
curl -X GET "http://localhost:5000/incidents/1?include_similar=true"

# Get incident without similar incidents
curl -X GET "http://localhost:5000/incidents/1?include_similar=false"
```

**Expected Response** (200 OK):
```json
{
  "incident": {
    "id": 1,
    "ticket_id": "TKT-001",
    "status": "resolved",
    "issue_type": "Login Issue",
    "product_area": "Authentication",
    "priority": "high",
    "initial_message": "Cannot login to my account...",
    "resolution_summary": "Reset password and cleared cache...",
    "created_at": "2026-01-15T10:20:00",
    "resolution_time_hours": 2.5,
    "customer_sentiment": "negative",
    "csat_score": 3,
    "platform": "Web",
    "region": "US-East"
  },
  "predictions": {
    "predicted_priority": "high",
    "confidence": 0.87,
    "all_probabilities": {
      "critical": 0.12,
      "high": 0.87,
      "medium": 0.01,
      "low": 0.00
    }
  },
  "similar_incidents": [
    {
      "id": 42,
      "ticket_id": "TKT-042",
      "status": "resolved",
      "issue_type": "Login Issue",
      "description": "Password reset not working...",
      "resolution": "Cleared browser cookies...",
      "scores": {
        "final": 0.8234,
        "similarity": 0.7823,
        "status": 1.0,
        "recency": 0.9102
      }
    }
  ]
}
```

**Test Cases**:
- ✅ Valid incident ID
- ❌ Non-existent ID (404 Not Found)
- ✅ ML prediction fallback when model fails

---

### **ENDPOINT 3: Search Similar Incidents** (GET /incidents/{id}/similar)

**Description**: Find top-N similar incidents using vector search

```bash
# Get top 5 similar incidents (default)
curl -X GET "http://localhost:5000/incidents/1/similar"

# Custom parameters
curl -X GET "http://localhost:5000/incidents/1/similar?top_k=50&top_n=10"
```

**Expected Response** (200 OK):
```json
{
  "incident_id": 1,
  "query": "Cannot login to my account, password reset not working",
  "total_candidates": 20,
  "results": [
    {
      "id": 523,
      "ticket_id": "TKT-523",
      "status": "resolved",
      "issue_type": "Authentication",
      "product_area": "User Management",
      "priority": "high",
      "description": "User unable to login, forgot password...",
      "resolution": "Sent password reset link via email...",
      "created_at": "2026-02-20T09:15:00",
      "sentiment": "neutral",
      "csat_score": 4,
      "scores": {
        "final": 0.8521,
        "similarity": 0.8123,
        "status": 1.0,
        "recency": 0.9234
      }
    }
  ]
}
```

**Test Cases**:
- ✅ Valid incident with message
- ❌ Incident without message (400 Bad Request)
- ❌ Invalid top_k or top_n range (400 Bad Request)

---

### **ENDPOINT 4: Resolve Incident with RAG** (POST /incidents/{id}/resolve)

**Description**: Generate AI-powered resolution using GPT-4 + vector search

```bash
# Resolve incident (first time)
curl -X POST "http://localhost:5000/incidents/100001/resolve" \
  -H "Content-Type: application/json" \
  -d '{
    "force_regenerate": false,
    "category": "network"
  }'

# Force regenerate resolution
curl -X POST "http://localhost:5000/incidents/1/resolve" \
  -H "Content-Type: application/json" \
  -d '{"force_regenerate": true}'
```

**Expected Response** (200 OK):
```json
{
  "incident_id": 100001,
  "ticket_id": "TKT-2026-0010",
  "query": "Unable to connect to VPN...",
  "answer": "Based on similar past incidents, here are the recommended troubleshooting steps:\n\n1. **Verify Network Connectivity**: Ensure your home network is stable and has internet access.\n\n2. **Check VPN Client Version**: Make sure you're using the latest VPN client (v5.2.1 or higher).\n\n3. **Firewall Settings**: Temporarily disable your firewall to see if it's blocking the VPN connection.\n\n4. **Clear VPN Cache**: Delete the VPN configuration cache:\n   - Windows: C:\\Users\\[YourName]\\AppData\\Local\\VPN\\cache\n   - Restart the VPN client\n\n5. **Reinstall VPN Client**: Uninstall and reinstall the VPN client if the issue persists.\n\n6. **Contact IT Support**: If none of the above steps work, escalate to IT support with error logs.",
  "source_incidents": [
    {
      "ticket_id": "TKT-4523",
      "issue_type": "VPN Access",
      "product_area": "Network Services",
      "description": "VPN timeout when connecting from home...",
      "resolution": "Updated VPN client to latest version, issue resolved",
      "similarity_score": 0.8234
    }
  ],
  "confidence": "high",
  "avg_similarity": 0.812,
  "resolution_stored": true
}
```

**Test Cases**:
- ✅ First-time resolution generation
- ❌ Duplicate resolution without force_regenerate (409 Conflict)
- ✅ Force regenerate existing resolution

---

### **ENDPOINT 5: Enhanced Health Check** (GET /health)

**Description**: Comprehensive system health check

```bash
curl -X GET "http://localhost:5000/health"
```

**Expected Response** (200 OK):
```json
{
  "status": "healthy",
  "service": "DecisionLens API",
  "version": "2.0.0",
  "timestamp": "2026-03-07T14:45:00",
  "database": {
    "connected": true,
    "incidents_count": 100000,
    "embeddings_count": 99523,
    "missing_embeddings": 477
  },
  "models": {
    "severity_model_loaded": true,
    "model_version": "v1",
    "model_path": "ml/models/severity_rf_v1.pkl"
  },
  "openai_api": "configured"
}
```

**Status Meanings**:
- `healthy`: All systems operational
- `degraded`: Some components unavailable
- `unhealthy`: Critical failure (503 response)

---

### **ENDPOINT 6: Trigger Model Retraining** (POST /batch/retrain)

**Description**: Start async model retraining job

```bash
curl -X POST "http://localhost:5000/batch/retrain" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "severity",
    "min_samples": 1000,
    "notify_email": "admin@company.com"
  }'
```

**Expected Response** (202 Accepted):
```json
{
  "job_id": "retrain_20260307_143022_abc123",
  "status": "started",
  "message": "Model retraining initiated with 100000 samples. Check /ml/status for progress.",
  "model_type": "severity",
  "estimated_duration": "5-10 minutes"
}
```

**Test Cases**:
- ✅ Valid retraining request
- ❌ Insufficient data (400 Bad Request)
- ❌ Invalid model_type (400 Bad Request)

---

### **ENDPOINT 7: ML Status and Metrics** (GET /ml/status)

**Description**: Get model metadata and embeddings statistics

```bash
curl -X GET "http://localhost:5000/ml/status"
```

**Expected Response** (200 OK):
```json
{
  "severity_model": {
    "loaded": true,
    "version": "v1",
    "path": "ml/models/severity_rf_v1.pkl",
    "last_trained": "2026-03-01T10:30:00",
    "accuracy": null
  },
  "embeddings": {
    "total_embeddings": 99523,
    "total_incidents": 100000,
    "coverage_percent": 99.52,
    "missing_count": 477,
    "last_generated": "2026-03-07T14:30:00",
    "model": "text-embedding-3-small",
    "dimensions": 1536
  },
  "openai_usage": {
    "api_key_configured": true,
    "models_used": ["text-embedding-3-small", "gpt-4"],
    "note": "Detailed usage tracking requires OpenAI API integration"
  }
}
```

---

## Performance Testing

### Load Test Example (using Apache Bench)

```bash
# Test health endpoint
ab -n 1000 -c 10 http://localhost:5000/health

# Test incident creation (requires JSON file)
ab -n 100 -c 5 -p incident.json -T application/json \
   http://localhost:5000/incidents
```

**Expected Performance**:
- Health check: <10ms
- Vector search: <50ms
- RAG resolution: 2-3 seconds (including GPT-4 call)

---

## Error Handling

### Common Error Responses

**400 Bad Request**:
```json
{
  "error": "top_k must be between 5 and 100",
  "timestamp": "2026-03-07T14:50:00"
}
```

**404 Not Found**:
```json
{
  "error": "Incident 999999 not found",
  "timestamp": "2026-03-07T14:51:00"
}
```

**409 Conflict**:
```json
{
  "error": "Incident with ticket_id 'TKT-001' already exists",
  "timestamp": "2026-03-07T14:52:00"
}
```

**422 Validation Error**:
```json
{
  "detail": [
    {
      "loc": ["body", "initial_message"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**500 Internal Server Error**:
```json
{
  "error": "Internal server error",
  "detail": "Database connection failed",
  "timestamp": "2026-03-07T14:53:00"
}
```

**503 Service Unavailable**:
```json
{
  "error": "Database connection failed: could not connect to server",
  "timestamp": "2026-03-07T14:54:00"
}
```

---

## Testing Workflow

### Full Integration Test

```bash
# 1. Check system health
curl http://localhost:5000/health

# 2. Check ML status
curl http://localhost:5000/ml/status

# 3. Create new incident
INCIDENT_ID=$(curl -X POST http://localhost:5000/incidents \
  -H "Content-Type: application/json" \
  -d '{"ticket_id":"TKT-TEST-001", "initial_message":"Test incident"}' \
  | jq -r '.incident_id')

# 4. Wait for embedding (2-3 seconds)
sleep 3

# 5. Get incident with predictions
curl http://localhost:5000/incidents/$INCIDENT_ID

# 6. Find similar incidents
curl http://localhost:5000/incidents/$INCIDENT_ID/similar

# 7. Generate resolution
curl -X POST http://localhost:5000/incidents/$INCIDENT_ID/resolve \
  -H "Content-Type: application/json" \
  -d '{"force_regenerate": false}'
```

---

## Monitoring Background Tasks

Background tasks (embeddings, retraining) run asynchronously. Check logs:

```bash
# Docker logs
docker logs -f decisionlens_api_dev

# Look for:
# [Background] Generating embedding for incident 100001
# [Background] ✓ Embedding stored for incident 100001
# [Background] Starting model retraining job retrain_20260307_143022_abc123
```

---

## Troubleshooting

### Common Issues

**Issue**: "Database connection failed"
- **Fix**: Ensure PostgreSQL is running: `docker-compose ps`

**Issue**: "Model file not found"
- **Fix**: Train the model first: `python ml/severity_model.py`

**Issue**: "OPENAI_API_KEY not configured"
- **Fix**: Add to `.env`: `OPENAI_API_KEY=sk-...`

**Issue**: Slow embedding generation
- **Fix**: Check OpenAI API rate limits and network latency

---

## Next Steps

1. **Production Deployment**:
   - Add authentication (OAuth2/JWT)
   - Configure proper CORS origins
   - Set up rate limiting
   - Add request logging and monitoring

2. **Enhanced Features**:
   - WebSocket support for real-time updates
   - Batch incident import endpoint
   - CSV export of incidents
   - Advanced search filters

3. **Testing**:
   - Add pytest unit tests
   - Integration tests with test database
   - Load testing with realistic data

4. **Documentation**:
   - API versioning strategy
   - Client SDK generation (OpenAPI)
   - Postman collection export
