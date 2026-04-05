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


class DuplicateReviewDecision(str, Enum):
    duplicate = "duplicate"
    related = "related"
    not_duplicate = "not_duplicate"


# ==================== REQUEST MODELS ====================

class CreateIncidentRequest(BaseModel):
    """Request body for POST /incidents"""
    initial_message: str = Field(..., min_length=10, max_length=10000, description="User's problem description")
    product_area: Optional[str] = Field(None, max_length=100)
    issue_type: Optional[str] = Field(None, max_length=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "initial_message": "Unable to access VPN from home network. Connection times out after authentication.",
                "product_area": "Network Services",
                "issue_type": "VPN Access",
            }
        }


class SearchIncidentsRequest(BaseModel):
    """Request body for GET /incidents/{id}/similar"""
    query: Optional[str] = Field(None, description="Override query text (defaults to incident description)")
    category: Optional[str] = Field(None, description="Filter by category")
    top_k: int = Field(20, ge=5, le=100, description="Number of candidates to fetch")
    top_n: int = Field(5, ge=1, le=20, description="Number of results to return")


class CheckDuplicatesRequest(BaseModel):
    """Request body for POST /incidents/check-duplicates"""
    initial_message: str = Field(..., min_length=10, max_length=10000, description="Problem description to compare")
    issue_type: Optional[str] = Field(None, max_length=100)
    product_area: Optional[str] = Field(None, max_length=100)
    top_k: int = Field(20, ge=5, le=100, description="Number of candidates to fetch")
    top_n: int = Field(5, ge=1, le=20, description="Number of results to return")

    class Config:
        json_schema_extra = {
            "example": {
                "initial_message": "Users cannot log in after the latest SSO rollout and password reset fails.",
                "issue_type": "SSO Login Failure",
                "product_area": "Identity Platform",
                "top_k": 20,
                "top_n": 5,
            }
        }


class SubmitDuplicateReviewRequest(BaseModel):
    """Request body for POST /duplicate-reviews"""
    reported_ticket_id: Optional[str] = Field(None, min_length=3, max_length=20)
    query_text: str = Field(..., min_length=10, max_length=10000, description="Original ticket text reviewed")
    matched_incident_id: int = Field(..., gt=0, description="Incident selected during duplicate review")
    decision: DuplicateReviewDecision = Field(..., description="Analyst review decision")
    issue_type: Optional[str] = Field(None, max_length=100)
    product_area: Optional[str] = Field(None, max_length=100)

    class Config:
        json_schema_extra = {
            "example": {
                "reported_ticket_id": "TKT-2026-1001",
                "query_text": "Users cannot log in after the latest SSO rollout and password reset fails.",
                "matched_incident_id": 42,
                "decision": "duplicate",
                "issue_type": "SSO Login Failure",
                "product_area": "Identity Platform",
            }
        }


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
    related_ticket_count: int = 1
    scores: SimilarityScores


class DuplicateCandidate(BaseModel):
    """Potential duplicate incident candidate"""
    id: int
    ticket_id: str
    status: Optional[str]
    issue_type: Optional[str]
    product_area: Optional[str]
    priority: Optional[str]
    description: str
    created_at: Optional[str]
    similarity_score: float = Field(..., ge=0, le=1)
    duplicate_score: float = Field(..., ge=0, le=1)
    related_ticket_count: int = 1
    match_reasons: List[str] = Field(default_factory=list)


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
    similar_incidents: List[SimilarIncident]


class SimilarIncidentsResponse(BaseModel):
    """Response for GET /incidents/{id}/similar"""
    incident_id: int
    query: str
    total_candidates: int
    results: List[SimilarIncident]


class CheckDuplicatesResponse(BaseModel):
    """Response for POST /incidents/check-duplicates"""
    query: str
    classification: str
    explanation: str
    total_candidates: int
    top_match_score: float
    results: List[DuplicateCandidate]


class DuplicateReviewResponse(BaseModel):
    """Response for POST /duplicate-reviews"""
    review_id: int
    status: str
    message: str


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


class HealthCheckResponse(BaseModel):
    """Enhanced health check response"""
    status: str
    service: str
    version: str
    timestamp: datetime
    database: DatabaseHealth
    openai_api: str


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
