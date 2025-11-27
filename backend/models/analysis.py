"""
Image Analysis and Processing Models

This module defines Pydantic models for image processing, AI model integration,
and analysis workflow management.
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid


class ImageFormat(str, Enum):
    """Supported image formats"""
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"


class AIModelType(str, Enum):
    """Available AI model types"""
    GROK = "grok"
    PLANTNET = "plantnet"
    INATURALIST = "inaturalist"
    CUSTOM = "custom"


class ProcessingStatus(str, Enum):
    """Image processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ImageMetadata(BaseModel):
    """Metadata about uploaded image"""
    filename: str = Field(..., description="Original filename")
    format: ImageFormat = Field(..., description="Image format")
    size_bytes: int = Field(..., description="File size in bytes")
    size_mb: float = Field(..., description="File size in MB")
    dimensions: Tuple[int, int] = Field(..., description="Image dimensions (width, height)")
    aspect_ratio: float = Field(..., description="Image aspect ratio")
    color_space: str = Field(..., description="Color space of image")
    bits_per_channel: Optional[int] = Field(None, description="Bits per color channel")

    @validator('size_mb')
    def validate_size_mb(cls, v):
        if v <= 0:
            raise ValueError('File size must be positive')
        return v

    @validator('dimensions')
    def validate_dimensions(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError('Invalid image dimensions')
        return v


class QualityMetrics(BaseModel):
    """Image quality assessment metrics"""
    overall_score: float = Field(..., description="Overall quality score (0-1)")
    sharpness_score: float = Field(..., description="Image sharpness (0-1)")
    brightness_score: float = Field(..., description="Brightness adequacy (0-1)")
    contrast_score: float = Field(..., description="Contrast quality (0-1)")
    noise_level: float = Field(..., description="Noise level (0-1, lower is better)")
    color_balance: float = Field(..., description="Color balance quality (0-1)")
    composition_score: float = Field(..., description="Image composition (0-1)")
    leaf_coverage: float = Field(..., description="Leaf area coverage (0-1)")

    @validator('overall_score')
    def validate_overall_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Overall quality score must be between 0 and 1')
        return v


class ImageAnalysisResult(BaseModel):
    """Results from image quality analysis"""
    metadata: ImageMetadata = Field(..., description="Image metadata")
    quality_metrics: QualityMetrics = Field(..., description="Quality assessment")
    is_acceptable: bool = Field(..., description="Whether image meets quality standards")
    processing_recommendations: List[str] = Field(..., description="Recommendations for improvement")
    preprocessing_applied: List[str] = Field(..., description="Preprocessing steps applied")


class AIModelConfig(BaseModel):
    """Configuration for AI model integration"""
    model_type: AIModelType = Field(..., description="Type of AI model")
    endpoint_url: str = Field(..., description="API endpoint URL")
    api_key_required: bool = Field(..., description="Whether API key is required")
    max_retries: int = Field(3, description="Maximum retry attempts")
    timeout_seconds: int = Field(30, description="Request timeout in seconds")
    confidence_threshold: float = Field(0.5, description="Minimum confidence threshold")
    enabled: bool = Field(True, description="Whether model is enabled")

    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence threshold must be between 0 and 1')
        return v


class PlantIdentification(BaseModel):
    """Plant identification result from AI model"""
    plant_name: str = Field(..., description="Identified plant name")
    confidence: float = Field(..., description="Confidence score (0-1)")
    scientific_name: Optional[str] = Field(None, description="Scientific name if available")
    alternative_identifications: List[Dict[str, Any]] = Field(..., description="Alternative possibilities")
    model_used: AIModelType = Field(..., description="AI model that made this identification")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")

    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v


class DiseaseIdentification(BaseModel):
    """Disease identification result from AI model"""
    disease_name: str = Field(..., description="Identified disease name")
    disease_id: Optional[str] = Field(None, description="Disease identifier if available")
    confidence: float = Field(..., description="Confidence score (0-1)")
    severity: str = Field(..., description="Severity level (mild/moderate/severe)")
    symptoms_detected: List[str] = Field(..., description="Symptoms observed in image")
    model_used: AIModelType = Field(..., description="AI model that made this identification")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")

    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v


class AIModelResult(BaseModel):
    """Complete result from a single AI model"""
    model_type: AIModelType = Field(..., description="Type of AI model")
    plant_identification: PlantIdentification = Field(..., description="Plant identification result")
    disease_identification: Optional[DiseaseIdentification] = Field(None, description="Disease identification result")
    raw_response: Optional[Dict[str, Any]] = Field(None, description="Raw API response")
    success: bool = Field(..., description="Whether the model call was successful")
    error_message: Optional[str] = Field(None, description="Error message if call failed")
    processing_time_ms: int = Field(..., description="Total processing time")
    timestamp: str = Field(..., description="Result timestamp")


class ModelAggregationWeights(BaseModel):
    """Weights for aggregating results from multiple models"""
    plant_identification: Dict[str, float] = Field(..., description="Weights for plant identification")
    disease_identification: Dict[str, float] = Field(..., description="Weights for disease identification")
    consensus_bonus: float = Field(0.1, description="Bonus for model agreement")

    @validator('consensus_bonus')
    def validate_consensus_bonus(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Consensus bonus must be between 0 and 1')
        return v


class ConfidenceAggregation(BaseModel):
    """Aggregated confidence scores from multiple models"""
    primary_plant_name: str = Field(..., description="Primary plant identification")
    plant_confidence: float = Field(..., description="Aggregated plant confidence")
    plant_alternatives: List[Dict[str, float]] = Field(..., description="Alternative plant identifications")
    primary_disease_name: Optional[str] = Field(None, description="Primary disease identification")
    disease_confidence: Optional[float] = Field(None, description="Aggregated disease confidence")
    disease_alternatives: List[Dict[str, float]] = Field(..., description="Alternative disease identifications")
    models_agreed: bool = Field(..., description="Whether models showed consensus")
    aggregation_method: str = Field(..., description="Method used for aggregation")


class ProcessingError(BaseModel):
    """Error details from processing pipeline"""
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    stage: str = Field(..., description="Processing stage where error occurred")
    model_type: Optional[AIModelType] = Field(None, description="Model that caused error if applicable")
    timestamp: str = Field(..., description="Error timestamp")
    retry_count: int = Field(..., description="Number of retry attempts made")


class AnalysisSession(BaseModel):
    """Complete analysis session tracking"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique session identifier")
    user_ip: Optional[str] = Field(None, description="User IP address for rate limiting")
    start_timestamp: str = Field(..., description="Session start time")
    end_timestamp: Optional[str] = Field(None, description="Session end time")
    status: ProcessingStatus = Field(..., description="Current processing status")
    total_processing_time_ms: int = Field(0, description="Total processing time")
    model_results: List[AIModelResult] = Field(..., description="Results from all models")
    aggregated_results: Optional[ConfidenceAggregation] = Field(None, description="Aggregated results")
    image_analysis: Optional[ImageAnalysisResult] = Field(None, description="Image quality analysis")
    errors: List[ProcessingError] = Field(..., description="Processing errors")
    warnings: List[str] = Field(..., description="Processing warnings")


class ImageProcessingRequest(BaseModel):
    """Request for image processing and analysis"""
    max_file_size_mb: Optional[float] = Field(10.0, description="Maximum file size in MB")
    min_quality_score: Optional[float] = Field(0.3, description="Minimum acceptable quality score")
    max_processing_time_ms: Optional[int] = Field(30000, description="Maximum processing time")
    enabled_models: Optional[List[AIModelType]] = Field(None, description="Models to use for analysis")
    include_alternatives: Optional[bool] = Field(True, description="Include alternative identifications")
    aggregation_weights: Optional[ModelAggregationWeights] = Field(None, description="Custom aggregation weights")

    class Config:
        use_enum_values = True


class ImageProcessingResponse(BaseModel):
    """Response from image processing and analysis"""
    success: bool = Field(..., description="Whether processing was successful")
    session_id: str = Field(..., description="Unique session identifier")
    status: ProcessingStatus = Field(..., description="Processing status")
    image_analysis: Optional[ImageAnalysisResult] = Field(None, description="Image quality analysis")
    results: Optional[Dict[str, Any]] = Field(None, description="Analysis results if successful")
    processing_time_ms: int = Field(..., description="Total processing time")
    timestamp: str = Field(..., description="Response timestamp")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    warnings: List[str] = Field(..., description="Processing warnings")
    model_performance: Dict[str, Any] = Field(..., description="Performance metrics by model")


class RateLimitInfo(BaseModel):
    """Rate limiting information"""
    requests_per_minute: int = Field(..., description="Allowed requests per minute")
    remaining_requests: int = Field(..., description="Remaining requests in current window")
    reset_timestamp: str = Field(..., description="When rate limit window resets")
    is_limited: bool = Field(..., description="Whether request is currently rate limited")


class APIResponseHeaders(BaseModel):
    """Standard API response headers"""
    request_id: str = Field(..., description="Unique request identifier")
    processing_time_ms: int = Field(..., description="Request processing time")
    rate_limit: Optional[RateLimitInfo] = Field(None, description="Rate limiting information")
    api_version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Response timestamp")


class BatchAnalysisRequest(BaseModel):
    """Request for batch analysis of multiple images"""
    max_concurrent_requests: Optional[int] = Field(5, description="Maximum concurrent analyses")
    timeout_per_image_ms: Optional[int] = Field(30000, description="Timeout per image")
    fail_fast: Optional[bool] = Field(True, description="Stop on first error")
    common_settings: Optional[ImageProcessingRequest] = Field(None, description="Common settings for all images")


class BatchAnalysisResponse(BaseModel):
    """Response from batch analysis"""
    batch_id: str = Field(..., description="Unique batch identifier")
    total_images: int = Field(..., description="Total number of images processed")
    successful_analyses: int = Field(..., description="Number of successful analyses")
    failed_analyses: int = Field(..., description="Number of failed analyses")
    total_processing_time_ms: int = Field(..., description="Total batch processing time")
    average_processing_time_ms: float = Field(..., description="Average processing time per image")
    individual_results: List[ImageProcessingResponse] = Field(..., description="Individual analysis results")
    errors: List[str] = Field(..., description="Batch-level errors")
    warnings: List[str] = Field(..., description="Batch-level warnings")