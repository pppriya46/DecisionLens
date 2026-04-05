"""
FastAPI Application for DecisionLens
RESTful API for incident intake, retrieval, and GPT-assisted troubleshooting
"""

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import psycopg2
import psycopg2.extras
import uuid
from dotenv import load_dotenv
from api.telemetry import emit_latency, now_ms, elapsed_ms

# Import models and dependencies
from api.models import (
    CreateIncidentRequest, CreateIncidentResponse,
    SimilarIncidentsResponse,
    ResolveIncidentRequest, ResolveIncidentResponse,
    GetIncidentResponse, HealthCheckResponse,
    CheckDuplicatesRequest, CheckDuplicatesResponse,
    SubmitDuplicateReviewRequest, DuplicateReviewResponse,
    IncidentDetail, SimilarIncident, DuplicateCandidate,
    DatabaseHealth
)
from api.dependencies import get_db_connection, verify_openai_key
from api.background_tasks import generate_embedding_task

# Import existing services
from api.search_service import search_similar_incidents, search_duplicate_incidents
from api.rag_service import generate_rag_response

load_dotenv()


def generate_ticket_id() -> str:
    return f"TKT-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"


def get_stored_incident_embedding(conn, incident_id: int):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT embedding_vector::text
            FROM incident_embeddings
            WHERE incident_id = %s
        """, (incident_id,))
        row = cur.fetchone()
    return row[0] if row else None

# ==================== APP INITIALIZATION ====================

app = FastAPI(
    title="DecisionLens API",
    description="Incident intelligence API with semantic search, duplicate-aware intake, and GPT-assisted troubleshooting",
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
            ticket_id = generate_ticket_id()
            while True:
                cur.execute("SELECT id FROM incidents WHERE ticket_id = %s", (ticket_id,))
                if not cur.fetchone():
                    break
                ticket_id = generate_ticket_id()
            
            cur.execute("""
                INSERT INTO incidents (
                    ticket_id, initial_message, product_area, issue_type, status, created_at
                )
                VALUES (%s, %s, %s, %s, %s, NOW())
                RETURNING id, ticket_id, status, created_at
            """, (
                ticket_id,
                incident.initial_message,
                incident.product_area,
                incident.issue_type,
                "open",
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


# ==================== ENDPOINT 2: GET INCIDENT ====================

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

        similar_incidents = []
        if include_similar and incident_data.get('initial_message'):
            try:
                stored_embedding = get_stored_incident_embedding(conn, incident_id)
                search_result = search_similar_incidents(
                    query_text=incident_data['initial_message'],
                    query_category=incident_data.get('issue_type'),
                    query_embedding=stored_embedding,
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
            similar_incidents=similar_incidents
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving incident: {str(e)}"
        )


@app.get("/incidents/by-ticket/{ticket_id}", response_model=GetIncidentResponse)
async def get_incident_by_ticket(
    ticket_id: str,
    include_similar: bool = False,
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
                WHERE ticket_id = %s
            """, (ticket_id,))

            incident_data = cur.fetchone()

            if not incident_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Ticket {ticket_id} not found"
                )

        incident = IncidentDetail(**incident_data)

        similar_incidents = []
        if include_similar and incident_data.get('initial_message'):
            try:
                stored_embedding = get_stored_incident_embedding(conn, incident_data['id'])
                search_result = search_similar_incidents(
                    query_text=incident_data['initial_message'],
                    query_category=incident_data.get('issue_type'),
                    query_embedding=stored_embedding,
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
            similar_incidents=similar_incidents
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving ticket: {str(e)}"
        )


# ==================== ENDPOINT 3: SEARCH SIMILAR INCIDENTS ====================

@app.get("/incidents/{incident_id}/similar", response_model=SimilarIncidentsResponse)
async def get_similar_incidents(
    incident_id: int,
    top_k: int = 20,
    top_n: int = 5,
    conn=Depends(get_db_connection)
):
    request_start_ms = now_ms()
    status_label = "success"
    search_timings = {}
    if top_k < 5 or top_k > 100:
        status_label = "validation_error"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="top_k must be between 5 and 100")
    
    if top_n < 1 or top_n > 20:
        status_label = "validation_error"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="top_n must be between 1 and 20")
    
    try:
        stored_embedding = None
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
        stored_embedding = get_stored_incident_embedding(conn, incident_id)
        
        search_result = search_similar_incidents(
            query_text=incident['initial_message'],
            query_category=incident.get('issue_type'),
            query_embedding=stored_embedding,
            top_k=top_k,
            top_n=top_n
        )
        search_timings = search_result.get('timings_ms', {})
        
        similar = [SimilarIncident(**inc) for inc in search_result['results']]
        
        return SimilarIncidentsResponse(
            incident_id=incident_id,
            query=search_result['query'],
            total_candidates=search_result['total_candidates'],
            results=similar
        )
    
    except HTTPException:
        status_label = "http_error"
        raise
    except Exception as e:
        status_label = "error"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error searching similar incidents: {str(e)}")
    finally:
        emit_latency(
            "api_request",
            endpoint="/incidents/{incident_id}/similar",
            incident_id=incident_id,
            top_k=top_k,
            top_n=top_n,
            status=status_label,
            embedding_source=search_result.get('embedding_source') if 'search_result' in locals() else None,
            search_timings_ms=search_timings,
            duration_ms=elapsed_ms(request_start_ms),
        )


# ==================== ENDPOINT 3B: CHECK DUPLICATES ====================

@app.post("/incidents/check-duplicates", response_model=CheckDuplicatesResponse)
async def check_duplicates(request: CheckDuplicatesRequest):
    request_start_ms = now_ms()
    status_label = "success"
    duplicate_timings = {}

    try:
        duplicate_result = search_duplicate_incidents(
            query_text=request.initial_message,
            query_issue_type=request.issue_type,
            query_product_area=request.product_area,
            top_k=request.top_k,
            top_n=request.top_n,
        )
        duplicate_timings = duplicate_result.get("timings_ms", {})

        return CheckDuplicatesResponse(
            query=duplicate_result["query"],
            classification=duplicate_result["classification"],
            explanation=duplicate_result["explanation"],
            total_candidates=duplicate_result["total_candidates"],
            top_match_score=duplicate_result["top_match_score"],
            results=[DuplicateCandidate(**item) for item in duplicate_result["results"]],
        )

    except HTTPException:
        status_label = "http_error"
        raise
    except Exception as e:
        status_label = "error"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking duplicates: {str(e)}"
        )
    finally:
        emit_latency(
            "api_request",
            endpoint="/incidents/check-duplicates",
            top_k=request.top_k,
            top_n=request.top_n,
            requested_issue_type=request.issue_type,
            requested_product_area=request.product_area,
            status=status_label,
            duplicate_timings_ms=duplicate_timings,
            duration_ms=elapsed_ms(request_start_ms),
        )


# ==================== ENDPOINT 3C: STORE DUPLICATE REVIEW ====================

@app.post("/duplicate-reviews", response_model=DuplicateReviewResponse, status_code=status.HTTP_201_CREATED)
async def submit_duplicate_review(
    request: SubmitDuplicateReviewRequest,
    conn=Depends(get_db_connection)
):
    request_start_ms = now_ms()
    status_label = "success"

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id FROM incidents WHERE id = %s", (request.matched_incident_id,))
            if not cur.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Matched incident {request.matched_incident_id} not found"
                )

            cur.execute("""
                INSERT INTO duplicate_reviews (
                    reported_ticket_id,
                    query_text,
                    matched_incident_id,
                    decision,
                    issue_type,
                    product_area
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                request.reported_ticket_id,
                request.query_text,
                request.matched_incident_id,
                request.decision.value,
                request.issue_type,
                request.product_area,
            ))
            review = cur.fetchone()
            conn.commit()

        return DuplicateReviewResponse(
            review_id=review["id"],
            status="stored",
            message=f"Stored '{request.decision.value}' review for incident {request.matched_incident_id}."
        )

    except HTTPException:
        status_label = "http_error"
        raise
    except psycopg2.Error as e:
        status_label = "db_error"
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        status_label = "error"
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error storing duplicate review: {str(e)}"
        )
    finally:
        emit_latency(
            "api_request",
            endpoint="/duplicate-reviews",
            matched_incident_id=request.matched_incident_id,
            decision=request.decision.value,
            status=status_label,
            duration_ms=elapsed_ms(request_start_ms),
        )


# ==================== ENDPOINT 4: RESOLVE INCIDENT WITH RAG ====================

@app.post("/incidents/{incident_id}/resolve", response_model=ResolveIncidentResponse)
async def resolve_incident(
    incident_id: int,
    request: ResolveIncidentRequest = ResolveIncidentRequest(),
    conn=Depends(get_db_connection)
):
    request_start_ms = now_ms()
    status_label = "success"
    rag_timings = {}
    try:
        stored_embedding = None
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
        stored_embedding = get_stored_incident_embedding(conn, incident_id)
        
        rag_result = generate_rag_response(
            query_text=incident['initial_message'],
            query_category=request.category or incident.get('issue_type'),
            query_embedding=stored_embedding,
        )
        rag_timings = rag_result.get('timings_ms', {})
        
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
        status_label = "http_error"
        raise
    except Exception as e:
        status_label = "error"
        conn.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error resolving incident: {str(e)}")
    finally:
        emit_latency(
            "api_request",
            endpoint="/incidents/{incident_id}/resolve",
            incident_id=incident_id,
            force_regenerate=request.force_regenerate,
            requested_category=request.category,
            status=status_label,
            embedding_source=rag_result.get('embedding_source') if 'rag_result' in locals() else None,
            rag_timings_ms=rag_timings,
            duration_ms=elapsed_ms(request_start_ms),
        )


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
            verify_openai_key()
            openai_status = "configured"
        except:
            openai_status = "not configured"
        
        overall_status = "healthy" if (
            database_health.connected and
            openai_status == "configured"
        ) else "degraded"
        
        return HealthCheckResponse(
            status=overall_status,
            service="DecisionLens API",
            version="2.0.0",
            timestamp=datetime.utcnow(),
            database=database_health,
            openai_api=openai_status
        )
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Health check failed: {str(e)}")


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
    print("\n" + "="*60)
    print("DecisionLens FastAPI v2.0.0")
    print("="*60)
    print("✓ Application started")
    print("✓ Endpoints: incident intake, search, duplicate review, and guided resolution")
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
