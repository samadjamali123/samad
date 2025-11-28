"""
Pydantic models for Plant Disease Detection API
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator, HttpUrl
from enum import Enum


class SeverityLevel(str, Enum):
    """Disease severity levels"""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"
    UNKNOWN = "Unknown"


class DiseaseCategory(str, Enum):
    """Disease categories"""
    FUNGAL = "Fungal"
    BACTERIAL = "Bacterial"
    VIRAL = "Viral"
    ENVIRONMENTAL = "Environmental"
    INSECT = "Insect"
    NUTRITIONAL = "Nutritional"
    PHYSICAL = "Physical"
    HEALTHY = "Healthy"


class DiseaseImage(BaseModel):
    """Disease image reference"""
    url: str = Field(..., description="URL to disease image")
    alt_text: str = Field(..., description="Alt text for image")
    image_type: str = Field(default="symptom", description="Type of image (symptom, prevention, treatment)")
    primary: bool = Field(default=False, description="Whether this is the primary image for the disease")


class Symptom(BaseModel):
    """Disease symptom description"""
    description: str = Field(..., description="Symptom description")
    severity: Optional[SeverityLevel] = Field(None, description="Symptom severity level")
    location: Optional[str] = Field(None, description="Where the symptom appears on the plant")
    visual_indicators: Optional[List[str]] = Field(None, description="Visual indicators to look for")


class Treatment(BaseModel):
    """Treatment recommendation"""
    description: str = Field(..., description="Treatment description")
    type: str = Field(..., description="Treatment type (chemical, organic, cultural)")
    effectiveness: Optional[str] = Field(None, description="Effectiveness level")
    application_method: Optional[str] = Field(None, description="How to apply the treatment")
    safety_notes: Optional[List[str]] = Field(None, description="Safety precautions")
    timing: Optional[str] = Field(None, description="When to apply the treatment")


class Prevention(BaseModel):
    """Prevention recommendation"""
    description: str = Field(..., description="Prevention measure description")
    effectiveness: Optional[str] = Field(None, description="Effectiveness level")
    implementation_difficulty: Optional[str] = Field(None, description="Difficulty level")
    frequency: Optional[str] = Field(None, description="How often to perform this prevention")
    cost: Optional[str] = Field(None, description="Relative cost level")


class Cause(BaseModel):
    """Disease cause information"""
    description: str = Field(..., description="Cause description")
    cause_type: str = Field(..., description="Type of cause (pathogen, environmental, nutritional)")
    scientific_name: Optional[str] = Field(None, description="Scientific name of cause")
    transmission: Optional[str] = Field(None, description="How the cause spreads")
    environmental_factors: Optional[List[str]] = Field(None, description="Environmental factors that contribute")


class Disease(BaseModel):
    """Comprehensive disease information model"""
    id: int = Field(..., ge=1, description="Unique disease identifier")
    name: str = Field(..., min_length=1, max_length=100, description="Common disease name")
    scientific_name: str = Field(..., min_length=1, max_length=200, description="Scientific name")
    category: DiseaseCategory = Field(..., description="Disease category")
    severity: SeverityLevel = Field(..., description="Overall disease severity")
    contagious: bool = Field(..., description="Whether the disease is contagious")
    affected_plants: List[str] = Field(..., min_items=1, description="List of affected plants")
    symptoms: List[Union[str, Symptom]] = Field(..., min_items=1, description="Disease symptoms")
    causes: List[Union[str, Cause]] = Field(..., min_items=1, description="Disease causes")
    treatment: List[Union[str, Treatment]] = Field(..., min_items=1, description="Treatment options")
    prevention: List[Union[str, Prevention]] = Field(..., min_items=1, description="Prevention measures")
    images: Optional[List[Union[str, DiseaseImage]]] = Field(None, description="Disease images")

    # Metadata
    created_at: Optional[datetime] = Field(None, description="When the disease was added to database")
    updated_at: Optional[datetime] = Field(None, description="When the disease was last updated")
    last_reviewed: Optional[datetime] = Field(None, description="When medical expert last reviewed")
    expert_verified: bool = Field(default=False, description="Whether verified by plant pathologist")
    source_reference: Optional[str] = Field(None, description="Source reference for information")
    notes: Optional[str] = Field(None, description="Additional notes about the disease")

    @validator('affected_plants')
    def validate_affected_plants(cls, v):
        if not v:
            raise ValueError('At least one affected plant must be specified')
        return [plant.strip() for plant in v if plant.strip()]

    @validator('symptoms')
    def validate_symptoms(cls, v):
        if not v:
            raise ValueError('At least one symptom must be specified')
        return v

    @validator('causes')
    def validate_causes(cls, v):
        if not v:
            raise ValueError('At least one cause must be specified')
        return v

    @validator('treatment')
    def validate_treatment(cls, v):
        if not v:
            raise ValueError('At least one treatment option must be specified')
        return v

    @validator('prevention')
    def validate_prevention(cls, v):
        if not v:
            raise ValueError('At least one prevention measure must be specified')
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        use_enum_values = True


class DiseaseCreate(BaseModel):
    """Model for creating new diseases (excludes auto-generated fields)"""
    name: str = Field(..., min_length=1, max_length=100)
    scientific_name: str = Field(..., min_length=1, max_length=200)
    category: DiseaseCategory
    severity: SeverityLevel
    contagious: bool
    affected_plants: List[str] = Field(..., min_items=1)
    symptoms: List[Union[str, Symptom]] = Field(..., min_items=1)
    causes: List[Union[str, Cause]] = Field(..., min_items=1)
    treatment: List[Union[str, Treatment]] = Field(..., min_items=1)
    prevention: List[Union[str, Prevention]] = Field(..., min_items=1)
    images: Optional[List[Union[str, DiseaseImage]]] = None
    source_reference: Optional[str] = None
    notes: Optional[str] = None

    class Config:
        use_enum_values = True


class DiseaseUpdate(BaseModel):
    """Model for updating existing diseases (all fields optional)"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    scientific_name: Optional[str] = Field(None, min_length=1, max_length=200)
    category: Optional[DiseaseCategory] = None
    severity: Optional[SeverityLevel] = None
    contagious: Optional[bool] = None
    affected_plants: Optional[List[str]] = Field(None, min_items=1)
    symptoms: Optional[List[Union[str, Symptom]]] = Field(None, min_items=1)
    causes: Optional[List[Union[str, Cause]]] = Field(None, min_items=1)
    treatment: Optional[List[Union[str, Treatment]]] = Field(None, min_items=1)
    prevention: Optional[List[Union[str, Prevention]] = Field(None, min_items=1)
    images: Optional[List[Union[str, DiseaseImage]]] = None
    last_reviewed: Optional[datetime] = None
    expert_verified: Optional[bool] = None
    source_reference: Optional[str] = None
    notes: Optional[str] = None

    class Config:
        use_enum_values = True


class DetectionPrediction(BaseModel):
    """Model for disease detection predictions"""
    disease_name: str = Field(..., description="Predicted disease name")
    scientific_name: str = Field(..., description="Scientific name of predicted disease")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    class_index: int = Field(..., ge=0, description="Model class index")
    severity: Optional[SeverityLevel] = Field(None, description="Predicted disease severity")
    top_k: Optional[List[Dict[str, Any]]] = Field(None, description="Top K predictions with probabilities")
    detected_at: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")

    class Config:
        use_enum_values = True


class ImageInfo(BaseModel):
    """Information about uploaded images"""
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., ge=0, description="File size in bytes")
    dimensions: List[int] = Field(..., min_items=2, max_items=2, description="Image dimensions [width, height]")
    format: str = Field(..., description="Image format (JPEG, PNG, etc.)")
    upload_time: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")
    processing_time: Optional[float] = Field(None, ge=0, description="Processing time in seconds")
    thumbnail_url: Optional[str] = Field(None, description="URL to thumbnail image")


class DetectionResult(BaseModel):
    """Complete detection result"""
    success: bool = Field(..., description="Whether detection was successful")
    prediction: Optional[DetectionPrediction] = Field(None, description="Detection prediction")
    image_info: ImageInfo = Field(..., description="Image information")
    processing_time: float = Field(..., ge=0, description="Total processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if detection failed")
    warnings: Optional[List[str]] = Field(None, description="Warning messages")
    model_version: Optional[str] = Field(None, description="Model version used")
    api_version: str = Field(default="1.0.0", description="API version used")

    class Config:
        use_enum_values = True


class BatchDetectionResult(BaseModel):
    """Results for batch image processing"""
    success: bool = Field(..., description="Overall batch success status")
    batch_info: Dict[str, Any] = Field(..., description="Batch processing information")
    results: List[Union[DetectionResult, Dict[str, Any]]] = Field(..., description="Individual results")
    total_processing_time: float = Field(..., ge=0, description="Total batch processing time")
    errors: Optional[List[str]] = Field(None, description="Batch-level errors")

    class Config:
        use_enum_values = True


class HealthStatus(BaseModel):
    """Health check status model"""
    success: bool = Field(..., description="Health check status")
    status: str = Field(..., description="Service status (healthy, degraded, unhealthy)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Status timestamp")
    service: str = Field(..., description="Service name")
    version: Optional[str] = Field(None, description="Service version")
    uptime: Optional[float] = Field(None, ge=0, description="Service uptime in seconds")
    system_info: Optional[Dict[str, Any]] = Field(None, description="System information")
    services: Optional[Dict[str, Any]] = Field(None, description="Individual service statuses")
    checks: Optional[Dict[str, Any]] = Field(None, description="Health check results")

    class Config:
        use_enum_values = True


class DatabaseStats(BaseModel):
    """Database statistics model"""
    total_diseases: int = Field(..., ge=0, description="Total number of diseases")
    total_categories: int = Field(..., ge=0, description="Total number of categories")
    total_severity_levels: int = Field(..., ge=0, description="Total number of severity levels")
    total_affected_plants: int = Field(..., ge=0, description="Total number of affected plants")
    contagious_diseases: int = Field(..., ge=0, description="Number of contagious diseases")
    expert_verified: int = Field(..., ge=0, description="Number of expert-verified diseases")
    categories: List[str] = Field(..., description="List of categories")
    severity_levels: List[str] = Field(..., description="List of severity levels")
    affected_plants: List[str] = Field(..., description="List of affected plants")
    database_last_updated: Optional[datetime] = Field(None, description="Database last update time")

    class Config:
        use_enum_values = True


class SearchQuery(BaseModel):
    """Search query model"""
    query: str = Field(..., min_length=2, description="Search query")
    category: Optional[DiseaseCategory] = Field(None, description="Filter by category")
    severity: Optional[SeverityLevel] = Field(None, description="Filter by severity")
    plant: Optional[str] = Field(None, description="Filter by affected plant")
    contagious: Optional[bool] = Field(None, description="Filter by contagious status")
    limit: Optional[int] = Field(10, ge=1, le=100, description="Maximum results")
    offset: Optional[int] = Field(0, ge=0, description="Results offset for pagination")
    sort_by: Optional[str] = Field("relevance", description="Sort field")
    sort_order: Optional[str] = Field("desc", description="Sort order (asc, desc)")

    class Config:
        use_enum_values = True


class SearchResult(BaseModel):
    """Search result model"""
    disease: Disease = Field(..., description="Disease information")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score")
    matched_fields: List[str] = Field(..., description="Fields that matched the search")
    excerpt: Optional[str] = Field(None, description="Relevant excerpt from disease description")

    class Config:
        use_enum_values = True


class SearchResults(BaseModel):
    """Search results collection model"""
    success: bool = Field(..., description="Search success status")
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., ge=0, description="Total number of results")
    results: List[SearchResult] = Field(..., description="Search results")
    facets: Optional[Dict[str, Any]] = Field(None, description="Search facets for filtering")
    search_time: float = Field(..., ge=0, description="Search time in seconds")
    suggestions: Optional[List[str]] = Field(None, description="Search suggestions")
    has_more: bool = Field(..., description="Whether more results are available")

    class Config:
        use_enum_values = True


# Response wrapper models for API consistency
class APIResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool = Field(..., description="Request success status")
    message: Optional[str] = Field(None, description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")

    class Config:
        use_enum_values = True


class ValidationError(BaseModel):
    """Validation error model"""
    field: str = Field(..., description="Field with validation error")
    message: str = Field(..., description="Validation error message")
    value: Optional[Any] = Field(None, description="Invalid value")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Detailed error response model"""
    success: bool = Field(default=False, description="Request success status")
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    validation_errors: Optional[List[ValidationError]] = Field(None, description="Validation error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    path: Optional[str] = Field(None, description="Request path")
    method: Optional[str] = Field(None, description="HTTP method")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")

    class Config:
        use_enum_values = True