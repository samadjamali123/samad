"""
Disease and Treatment Data Models

This module defines Pydantic models for diseases, treatments, and related data structures.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum


class SeverityLevel(str, Enum):
    """Disease severity classification"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class TreatmentPriority(str, Enum):
    """Treatment priority levels"""
    ORGANIC = "organic"
    CHEMICAL = "chemical"
    PREVENTION = "prevention"
    PROFESSIONAL = "professional"


class TreatmentMethod(BaseModel):
    """Individual treatment method details"""
    name: str = Field(..., description="Name of the treatment method")
    application: str = Field(..., description="How to apply the treatment")
    dosage: str = Field(..., description="Dosage or amount to use")
    safety: str = Field(..., description="Safety precautions")
    waiting_period: Optional[str] = Field(None, description="Days to wait before harvest")
    effectiveness: str = Field(..., description="Effectiveness percentage")
    cost_range: str = Field(..., description="Cost range estimate")
    environmental_impact: Optional[str] = Field(None, description="Environmental considerations")


class TreatmentCategory(BaseModel):
    """Category of treatments with methods and practices"""
    priority: int = Field(..., description="Priority level (1=highest)")
    methods: List[TreatmentMethod] = Field(..., description="List of treatment methods")
    cultural_practices: Optional[List[str]] = Field(None, description="Cultural practices to follow")
    effectiveness_summary: str = Field(..., description="Overall effectiveness summary")


class ProfessionalConsultation(BaseModel):
    """Professional consultation recommendations"""
    when_to_consult: List[str] = Field(..., description="When to seek professional help")
    contact_resources: List[str] = Field(..., description="Who to contact for help")
    yield_loss_thresholds: Optional[List[str]] = Field(None, description="Yield loss indicators")


class EnvironmentalFactors(BaseModel):
    """Environmental factors affecting disease development"""
    temperature_range: str = Field(..., description="Optimal temperature range for disease")
    humidity_required: str = Field(..., description="Required humidity level")
    susceptible_stages: str = Field(..., description="Plant stages most susceptible")
    other_factors: Optional[List[str]] = Field(None, description="Other contributing factors")


class SeverityClassification(BaseModel):
    """Severity level descriptions and thresholds"""
    mild: str = Field(..., description="Description of mild symptoms")
    moderate: str = Field(..., description="Description of moderate symptoms")
    severe: str = Field(..., description="Description of severe symptoms")
    percentage_thresholds: Optional[Dict[str, str]] = Field(None, description="Leaf area percentage thresholds")


class Disease(BaseModel):
    """Comprehensive disease information model"""
    disease_id: str = Field(..., description="Unique disease identifier")
    plant_name: str = Field(..., description="Common name of the plant")
    disease_name: str = Field(..., description="Common name of the disease")
    scientific_name: str = Field(..., description="Scientific name of the pathogen")
    description: str = Field(..., description="Detailed description of the disease")
    symptoms: List[str] = Field(..., description="List of visible symptoms")
    causes: List[str] = Field(..., description="Primary causes of the disease")
    spread_methods: List[str] = Field(..., description="How the disease spreads")
    environmental_factors: EnvironmentalFactors = Field(..., description="Environmental conditions")
    severity_levels: SeverityClassification = Field(..., description="Severity descriptions")
    reference_images: List[str] = Field(..., description="Paths to reference images")

    # Treatment information (loaded separately)
    treatments: Optional[Dict[str, Any]] = Field(None, description="Treatment recommendations")


class TreatmentInfo(BaseModel):
    """Complete treatment information for a disease"""
    disease_id: str = Field(..., description="Disease identifier")
    organic: TreatmentCategory = Field(..., description="Organic treatment methods")
    chemical: TreatmentCategory = Field(..., description="Chemical treatment methods")
    prevention: Dict[str, str] = Field(..., description="Prevention strategies")
    professional_consultation: ProfessionalConsultation = Field(..., description="When to seek professional help")


class PlantDetection(BaseModel):
    """Plant identification results"""
    plant_name: str = Field(..., description="Identified plant name")
    confidence: float = Field(..., description="Confidence score (0-1)")
    alternative_plants: List[Dict[str, float]] = Field(..., description="Alternative plant possibilities")

    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v


class DiseaseDetection(BaseModel):
    """Disease identification results"""
    primary_disease: Dict[str, Any] = Field(..., description="Primary detected disease")
    secondary_diseases: List[Dict[str, Any]] = Field(..., description="Secondary disease possibilities")

    @validator('primary_disease')
    def validate_primary_disease(cls, v):
        required_fields = ['disease_id', 'disease_name', 'confidence', 'severity']
        for field in required_fields:
            if field not in v:
                raise ValueError(f'Primary disease must contain {field}')
        return v


class AIModelResults(BaseModel):
    """Results from individual AI models"""
    grok_analysis: Dict[str, Any] = Field(..., description="Grok AI model results")
    specialized_model: Dict[str, Any] = Field(..., description="Specialized plant disease model results")
    other_models: Optional[List[Dict[str, Any]]] = Field(None, description="Additional model results")


class AnalysisMetadata(BaseModel):
    """Metadata about the analysis process"""
    image_quality_score: float = Field(..., description="Quality assessment of uploaded image")
    processing_time_total_ms: int = Field(..., description="Total processing time in milliseconds")
    models_used: List[str] = Field(..., description="List of AI models used")
    analysis_timestamp: str = Field(..., description="Timestamp of analysis")

    @validator('image_quality_score')
    def validate_quality_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Image quality score must be between 0 and 1')
        return v


class AnalysisResults(BaseModel):
    """Complete analysis results from AI models"""
    plant_detection: PlantDetection = Field(..., description="Plant identification results")
    disease_detection: DiseaseDetection = Field(..., description="Disease identification results")
    ai_model_results: AIModelResults = Field(..., description="Individual AI model results")
    metadata: AnalysisMetadata = Field(..., description="Analysis metadata")


class AnalysisRequest(BaseModel):
    """Request model for image analysis"""
    image_quality_threshold: Optional[float] = Field(0.3, description="Minimum image quality threshold")
    include_alternatives: Optional[bool] = Field(True, description="Include alternative identifications")
    max_processing_time_ms: Optional[int] = Field(30000, description="Maximum processing time")


class AnalysisResponse(BaseModel):
    """Response model for image analysis API"""
    success: bool = Field(..., description="Whether analysis was successful")
    analysis_id: str = Field(..., description="Unique identifier for this analysis")
    timestamp: str = Field(..., description="Analysis timestamp")
    results: Optional[AnalysisResults] = Field(None, description="Analysis results if successful")
    error_message: Optional[str] = Field(None, description="Error message if analysis failed")
    warnings: Optional[List[str]] = Field(None, description="Any warnings about the analysis")


class DiseaseInfoRequest(BaseModel):
    """Request model for disease information"""
    include_treatments: Optional[bool] = Field(True, description="Include treatment recommendations")
    include_images: Optional[bool] = Field(True, description="Include reference image paths")


class DiseaseInfoResponse(BaseModel):
    """Response model for disease information API"""
    success: bool = Field(..., description="Whether request was successful")
    disease: Optional[Disease] = Field(None, description="Disease information if successful")
    treatments: Optional[TreatmentInfo] = Field(None, description="Treatment information if requested")
    error_message: Optional[str] = Field(None, description="Error message if request failed")


class HealthCheckResponse(BaseModel):
    """Response model for health check API"""
    status: str = Field(..., description="Service health status")
    timestamp: str = Field(..., description="Health check timestamp")
    services: Dict[str, bool] = Field(..., description="Status of external services")
    version: str = Field(..., description="API version")
    uptime_seconds: Optional[int] = Field(None, description="Service uptime in seconds")


class ImageValidationResult(BaseModel):
    """Result of image validation"""
    is_valid: bool = Field(..., description="Whether image passed validation")
    format: str = Field(..., description="Detected image format")
    size_mb: float = Field(..., description="File size in MB")
    dimensions: tuple = Field(..., description="Image dimensions (width, height)")
    quality_score: Optional[float] = Field(None, description="Quality assessment score")
    errors: List[str] = Field(..., description="Validation error messages")
    warnings: List[str] = Field(..., description="Validation warning messages")


class DatabaseStats(BaseModel):
    """Database statistics for health checks"""
    total_diseases: int = Field(..., description="Total number of diseases in database")
    total_treatments: int = Field(..., description="Total number of treatment entries")
    total_reference_images: int = Field(..., description="Total reference images")
    last_updated: str = Field(..., description="Last database update timestamp")
    supported_plants: List[str] = Field(..., description="List of supported plants")