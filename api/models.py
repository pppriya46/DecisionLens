"""
Pydantic Models for DecisionLens FastAPI
Request/Response schemas with validation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ==================== ENUMS ====================

class IncidentStatus(str, Enum):
    open = "open"
    in_progress = "in_progress"
    on_hold = "on_hold"
    resolved = "resolved"
    closed = "closed"


class IncidentPriority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class CustomerSentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


# ==================== REQUEST MODELS ====================

class CreateIncidentRequest(BaseModel):
    """Request body for POST /incidents"""
    ticket_id: str = Field(..., min_length=3, max_length=20, description="Unique ticket identifier")
    initial_message: str = Field(..., min_length=10, max_length=10000, description="User's problem description")
    customer_id: Optional[str] = Field(None, max_length=20)
    customer_segment: Optional[str] = Field(None, max_length=50)
    channel: Optional[str] = Field("web", max_length=50)
    product_area: Optional[str] = Field(None, max_length=100)
    issue_type: Optional[str] = Field(None, max_length=100)
    priority: Optional[IncidentPriority] = Field(IncidentPriority.medium)
    platform: Optional[str] = Field(None, max_length=50)
    region: Optional[str] = Field(None, max_length=20)
    has_attachment: Optional[bool] = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "TKT-2026-0001",
                "initial_message": "Unable to access VPN from home network. Connection times out after authentication.",
                "customer_id": "CUST-5432",
                "channel": "email",
                "product_area": "Network Services",
                "issue_type": "VPN Access",
                "priority": "high",
                "platform": "Windows 11",
                "region": "US-West"
            }
        }


class SearchIncidentsRequest(BaseModel):
    """Request body for GET /incidents/{id}/similar"""
    query: Optional[str] = Field(None, description="Override query text (defaults to incident description)")
    category: Optional[str] = Field(None, description="Filter by category")
    top_k: int = Field(20, ge=5, le=100, description="Number of candidates to fetch")
    top_n: int = Field(5, ge=1, le=20, description="Number of results to return")


class ResolveIncidentRequest(BaseModel):
    """Request body for POST /incidents/{id}/resolve"""
    force_regenerate: bool = Field(False, description="Force regenerate even if resolution exists")
    category: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "force_regenerate": False,
                "category": "network"
            }
        }


class RetrainModelRequest(BaseModel):
    """Request body for POST /batch/retrain"""
    model_type: str = Field("severity", description="Model to retrain (currently only 'severity')")
    min_samples: int = Field(1000, ge=100, description="Minimum samples required for retraining")
    notify_email: Optional[str] = Field(None, description="Email to notify when complete")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "severity",
                "min_samples": 1000,
                "notify_email": "admin@company.com"
            }
        }


# ==================== RESPONSE MODELS ====================

class SimilarityScores(BaseModel):
    """Breakdown of similarity scoring"""
    final: float = Field(..., ge=0, le=1)
    similarity: float = Field(..., ge=0, le=1)
    status: float = Field(..., ge=0, le=1)
    recency: float = Field(..., ge=0, le=1)


class SimilarIncident(BaseModel):
    """Similar incident with scores"""
    id: int
    ticket_id: str
    status: str
    issue_type: Optional[str]
    product_area: Optional[str]
    priority: Optional[str]
    description: str
    resolution: Optional[str]
    created_at: Optional[str]
    sentiment: Optional[str]
    csat_score: Optional[int]
    scores: SimilarityScores


class SeverityPrediction(BaseModel):
    """ML severity prediction"""
    predicted_priority: str = Field(..., description="Predicted priority level")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    all_probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")


class IncidentDetail(BaseModel):
    """Full incident details"""
    id: int
    ticket_id: str
    status: str
    issue_type: Optional[str]
    product_area: Optional[str]
    priority: Optional[str]
    initial_message: str
    resolution_summary: Optional[str]
    created_at: Optional[datetime]
    resolution_time_hours: Optional[float]
    customer_sentiment: Optional[str]
    csat_score: Optional[int]
    platform: Optional[str]
    region: Optional[str]


class CreateIncidentResponse(BaseModel):
    """Response for POST /incidents"""
    incident_id: int
    ticket_id: str
    status: str = "open"
    message: str = "Incident created successfully. Embedding generation in progress."
    created_at: datetime


class GetIncidentResponse(BaseModel):
    """Response for GET /incidents/{id}"""
    incident: IncidentDetail
    predictions: SeverityPrediction
    similar_incidents: List[SimilarIncident]


class SimilarIncidentsResponse(BaseModel):
    """Response for GET /incidents/{id}/similar"""
    incident_id: int
    query: str
    total_candidates: int
    results: List[SimilarIncident]


class SourceIncident(BaseModel):
    """Source incident for RAG response"""
    ticket_id: str
    issue_type: Optional[str]
    product_area: Optional[str]
    description: str
    resolution: Optional[str]
    similarity_score: float


class ResolveIncidentResponse(BaseModel):
    """Response for POST /incidents/{id}/resolve"""
    incident_id: int
    ticket_id: str
    query: str
    answer: str
    source_incidents: List[SourceIncident]
    confidence: str
    avg_similarity: float
    resolution_stored: bool


class DatabaseHealth(BaseModel):
    """Database health status"""
    connected: bool
    incidents_count: int
    embeddings_count: int
    missing_embeddings: int


class ModelHealth(BaseModel):
    """Model health status"""
    severity_model_loaded: bool
    model_version: str
    model_path: str


class HealthCheckResponse(BaseModel):
    """Enhanced health check response"""
    status: str
    service: str
    version: str
    timestamp: datetime
    database: DatabaseHealth
    models: ModelHealth
    openai_api: str


class ModelMetrics(BaseModel):
    """Model training metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


class MLStatusResponse(BaseModel):
    """Response for GET /ml/status"""
    severity_model: Dict[str, Any]
    embeddings: Dict[str, Any]
    openai_usage: Dict[str, Any]


class RetrainResponse(BaseModel):
    """Response for POST /batch/retrain"""
    job_id: str
    status: str
    message: str
    model_type: str
    estimated_duration: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "retrain_20260307_143022_abc123",
                "status": "started",
                "message": "Model retraining initiated. Check /ml/status for progress.",
                "model_type": "severity",
                "estimated_duration": "5-10 minutes"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
