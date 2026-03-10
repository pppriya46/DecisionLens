"""
FastAPI Application - DecisionLens Phase 2
RESTful API with RAG, Vector Search, and ML Integration
"""

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import os
import joblib
import psycopg2
import psycopg2.extras
import uuid
from dotenv import load_dotenv

# Import models and dependencies
from api.models import (
    CreateIncidentRequest, CreateIncidentResponse,
    SearchIncidentsRequest, SimilarIncidentsResponse,
    ResolveIncidentRequest, ResolveIncidentResponse,
    GetIncidentResponse, HealthCheckResponse,
    MLStatusResponse, RetrainModelRequest, RetrainResponse,
    IncidentDetail, SeverityPrediction, SimilarIncident,
    DatabaseHealth, ModelHealth, ErrorResponse
)
from api.dependencies import get_db_connection, verify_openai_key, get_model_path
from api.background_tasks import generate_embedding_task, retrain_severity_model_task

# Import existing services
from api.search_service import search_similar_incidents
from api.rag_service import generate_rag_response
from ml.predict_severity import predict_severity

load_dotenv()

# Global model variables
severity_model    = None
severity_encoders = None
severity_tfidf    = None

# ==================== APP INITIALIZATION ====================

app = FastAPI(
    title="DecisionLens API",
    description="AI-powered IT Support RAG System with Vector Search and ML Predictions",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== EXCEPTION HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ==================== ENDPOINT 1: CREATE INCIDENT ====================

@app.post("/incidents", response_model=CreateIncidentResponse, status_code=status.HTTP_201_CREATED)
async def create_incident(
    incident: CreateIncidentRequest,
    background_tasks: BackgroundTasks,
    conn=Depends(get_db_connection)
):
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id FROM incidents WHERE ticket_id = %s", (incident.ticket_id,))
            if cur.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Incident with ticket_id '{incident.ticket_id}' already exists"
                )
            
            cur.execute("""
                INSERT INTO incidents (
                    ticket_id, initial_message, customer_id, customer_segment,
                    channel, product_area, issue_type, priority, status,
                    platform, region, has_attachment, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                RETURNING id, ticket_id, status, created_at
            """, (
                incident.ticket_id,
                incident.initial_message,
                incident.customer_id,
                incident.customer_segment,
                incident.channel,
                incident.product_area,
                incident.issue_type,
                incident.priority.value if incident.priority else "medium",
                "open",
                incident.platform,
                incident.region,
                incident.has_attachment
            ))
            
            result = cur.fetchone()
            conn.commit()
            incident_id = result['id']
            background_tasks.add_task(generate_embedding_task, incident_id)
            
            return CreateIncidentResponse(
                incident_id=incident_id,
                ticket_id=result['ticket_id'],
                status=result['status'],
                message="Incident created successfully. Embedding generation in progress.",
                created_at=result['created_at']
            )
    
    except psycopg2.Error as e:
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )


# ==================== ENDPOINT 2: GET INCIDENT WITH PREDICTIONS ====================

@app.get("/incidents/{incident_id}", response_model=GetIncidentResponse)
async def get_incident(
    incident_id: int,
    include_similar: bool = True,
    conn=Depends(get_db_connection)
):
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    id, ticket_id, status, issue_type, product_area, priority,
                    initial_message, resolution_summary, created_at,
                    resolution_time_hours, customer_sentiment, csat_score,
                    platform, region
                FROM incidents
                WHERE id = %s
            """, (incident_id,))
            
            incident_data = cur.fetchone()
            
            if not incident_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Incident {incident_id} not found"
                )
        
        incident = IncidentDetail(**incident_data)
        
        try:
            prediction_result = predict_severity(
                category=incident_data.get('issue_type', 'Unknown'),
                subcategory=incident_data.get('product_area', 'Unknown'),
                contact_type="Web",
                reassignment_count=0,
                reopen_count=0,
                sys_mod_count=1,
                made_sla=True,
                knowledge=False,
                initial_message=incident_data.get('initial_message', ''),
                customer_sentiment=incident_data.get('customer_sentiment', 'neutral') or 'neutral',
            )
            predictions = SeverityPrediction(**prediction_result)
        
        except Exception as e:
            print(f"[Warning] Severity prediction failed: {e}")
            predictions = SeverityPrediction(
                predicted_priority="medium",
                confidence=0.0,
                all_probabilities={}
            )
        
        similar_incidents = []
        if include_similar and incident_data.get('initial_message'):
            try:
                search_result = search_similar_incidents(
                    query_text=incident_data['initial_message'],
                    query_category=incident_data.get('issue_type'),
                    top_k=20,
                    top_n=5
                )
                similar_incidents = [
                    SimilarIncident(**inc) for inc in search_result.get('results', [])
                ]
            except Exception as e:
                print(f"[Warning] Similar incidents search failed: {e}")
        
        return GetIncidentResponse(
            incident=incident,
            predictions=predictions,
            similar_incidents=similar_incidents
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving incident: {str(e)}"
        )


# ==================== ENDPOINT 3: SEARCH SIMILAR INCIDENTS ====================

@app.get("/incidents/{incident_id}/similar", response_model=SimilarIncidentsResponse)
async def get_similar_incidents(
    incident_id: int,
    top_k: int = 20,
    top_n: int = 5,
    conn=Depends(get_db_connection)
):
    if top_k < 5 or top_k > 100:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="top_k must be between 5 and 100")
    
    if top_n < 1 or top_n > 20:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="top_n must be between 1 and 20")
    
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, initial_message, issue_type
                FROM incidents WHERE id = %s
            """, (incident_id,))
            
            incident = cur.fetchone()
            
            if not incident:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Incident {incident_id} not found")
            
            if not incident['initial_message']:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incident has no message to search with")
        
        search_result = search_similar_incidents(
            query_text=incident['initial_message'],
            query_category=incident.get('issue_type'),
            top_k=top_k,
            top_n=top_n
        )
        
        similar = [SimilarIncident(**inc) for inc in search_result['results']]
        
        return SimilarIncidentsResponse(
            incident_id=incident_id,
            query=search_result['query'],
            total_candidates=search_result['total_candidates'],
            results=similar
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error searching similar incidents: {str(e)}")


# ==================== ENDPOINT 4: RESOLVE INCIDENT WITH RAG ====================

@app.post("/incidents/{incident_id}/resolve", response_model=ResolveIncidentResponse)
async def resolve_incident(
    incident_id: int,
    request: ResolveIncidentRequest = ResolveIncidentRequest(),
    conn=Depends(get_db_connection)
):
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, ticket_id, initial_message, issue_type, 
                       resolution_summary, status
                FROM incidents WHERE id = %s
            """, (incident_id,))
            
            incident = cur.fetchone()
            
            if not incident:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Incident {incident_id} not found")
            
            if incident['resolution_summary'] and not request.force_regenerate:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Incident already has a resolution. Use force_regenerate=true to override."
                )
        
        rag_result = generate_rag_response(
            query_text=incident['initial_message'],
            query_category=request.category or incident.get('issue_type')
        )
        
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE incidents SET resolution_summary = %s, status = 'resolved'
                WHERE id = %s
            """, (rag_result['answer'], incident_id))
            conn.commit()
        
        return ResolveIncidentResponse(
            incident_id=incident_id,
            ticket_id=incident['ticket_id'],
            query=rag_result['query'],
            answer=rag_result['answer'],
            source_incidents=rag_result['source_incidents'],
            confidence=rag_result['confidence'],
            avg_similarity=rag_result['avg_similarity'],
            resolution_stored=True
        )
    
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error resolving incident: {str(e)}")


# ==================== ENDPOINT 5: HEALTH CHECK ====================

@app.get("/health", response_model=HealthCheckResponse)
async def health_check(conn=Depends(get_db_connection)):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM incidents")
            incidents_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM incident_embeddings")
            embeddings_count = cur.fetchone()[0]
            
            cur.execute("""
                SELECT COUNT(*) FROM incidents i
                LEFT JOIN incident_embeddings ie ON i.id = ie.incident_id
                WHERE ie.id IS NULL
            """)
            missing_embeddings = cur.fetchone()[0]
        
        database_health = DatabaseHealth(
            connected=True,
            incidents_count=incidents_count,
            embeddings_count=embeddings_count,
            missing_embeddings=missing_embeddings
        )
        
        try:
            model_path = get_model_path("severity")
            model_health = ModelHealth(severity_model_loaded=True, model_version="v1", model_path=model_path)
        except:
            model_health = ModelHealth(severity_model_loaded=False, model_version="unknown", model_path="not found")
        
        try:
            verify_openai_key()
            openai_status = "configured"
        except:
            openai_status = "not configured"
        
        overall_status = "healthy" if (
            database_health.connected and
            model_health.severity_model_loaded and
            openai_status == "configured"
        ) else "degraded"
        
        return HealthCheckResponse(
            status=overall_status,
            service="DecisionLens API",
            version="2.0.0",
            timestamp=datetime.utcnow(),
            database=database_health,
            models=model_health,
            openai_api=openai_status
        )
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Health check failed: {str(e)}")


# ==================== ENDPOINT 6: TRIGGER MODEL RETRAINING ====================

@app.post("/batch/retrain", response_model=RetrainResponse, status_code=status.HTTP_202_ACCEPTED)
async def retrain_model(
    request: RetrainModelRequest,
    background_tasks: BackgroundTasks,
    conn=Depends(get_db_connection)
):
    if request.model_type != "severity":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only 'severity' model is supported")
    
    job_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM incidents WHERE priority IS NOT NULL")
        sample_count = cur.fetchone()[0]
    
    if sample_count < request.min_samples:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Insufficient samples: {sample_count} < {request.min_samples}")
    
    background_tasks.add_task(retrain_severity_model_task, job_id, request.min_samples)
    
    return RetrainResponse(
        job_id=job_id,
        status="started",
        message=f"Model retraining initiated with {sample_count} samples. Check /ml/status for progress.",
        model_type=request.model_type,
        estimated_duration="5-10 minutes"
    )


# ==================== ENDPOINT 7: ML STATUS ====================

@app.get("/ml/status", response_model=MLStatusResponse)
async def get_ml_status(conn=Depends(get_db_connection)):
    try:
        model_info = {
            "loaded": False,
            "version": "unknown",
            "path": "ml/models/severity_rf_v1.pkl",
            "last_trained": None,
            "accuracy": None
        }
        
        try:
            model_path = get_model_path("severity")
            model_info["loaded"] = True
            model_info["version"] = "v1"
            model_mtime = os.path.getmtime(model_path)
            model_info["last_trained"] = datetime.fromtimestamp(model_mtime).isoformat()
        except:
            pass
        
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM incident_embeddings")
            total_embeddings = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM incidents")
            total_incidents = cur.fetchone()[0]
            
            cur.execute("SELECT MAX(created_at) FROM incident_embeddings")
            last_generated = cur.fetchone()[0]
        
        embeddings_coverage = (total_embeddings / total_incidents * 100) if total_incidents > 0 else 0
        
        embeddings_info = {
            "total_embeddings": total_embeddings,
            "total_incidents": total_incidents,
            "coverage_percent": round(embeddings_coverage, 2),
            "missing_count": total_incidents - total_embeddings,
            "last_generated": last_generated.isoformat() if last_generated else None,
            "model": "text-embedding-3-small",
            "dimensions": 1536
        }
        
        openai_info = {
            "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
            "models_used": ["text-embedding-3-small", "gpt-4"],
            "note": "Detailed usage tracking requires OpenAI API integration"
        }
        
        return MLStatusResponse(
            severity_model=model_info,
            embeddings=embeddings_info,
            openai_usage=openai_info
        )
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error retrieving ML status: {str(e)}")


# ==================== ROOT ====================

@app.get("/")
async def root():
    return {
        "service": "DecisionLens API",
        "version": "2.0.0",
        "description": "AI-powered IT Support RAG System",
        "documentation": "/docs",
        "health_check": "/health"
    }


# ==================== STARTUP EVENT ====================

@app.on_event("startup")
async def startup_event():
    global severity_model, severity_encoders, severity_tfidf
    print("\n" + "="*60)
    print("DecisionLens FastAPI v2.0.0")
    print("="*60)
    print("✓ Application started")

    try:
        model_path   = os.getenv("SEVERITY_MODEL_PATH", "ml/models/severity_rf_v1.pkl")
        encoder_path = os.getenv("ENCODER_PATH", "ml/models/label_encoders.pkl")
        tfidf_path   = os.getenv("TFIDF_PATH", "ml/models/tfidf_vectorizer.pkl")

        severity_model    = joblib.load(model_path)
        severity_encoders = joblib.load(encoder_path)
        severity_tfidf    = joblib.load(tfidf_path)
        print(f"✓ Severity model loaded from {model_path}")
    except Exception as e:
        print(f"✗ Failed to load severity model: {e}")

    print("✓ Endpoints: 7 RESTful routes available")
    print("✓ Documentation: http://localhost:5000/docs")
    print("✓ Health check: http://localhost:5000/health")
    print("="*60 + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        reload=True
    )