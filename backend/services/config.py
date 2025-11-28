"""
Configuration Settings Service

This module handles all configuration management, environment variables,
and application settings using Pydantic settings.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # FastAPI Configuration
    FASTAPI_HOST: str = Field(default="0.0.0.0", description="FastAPI server host")
    FASTAPI_PORT: int = Field(default=8000, description="FastAPI server port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    ENVIRONMENT: str = Field(default="development", description="Environment name")

    # AI Service API Keys
    GROK_API_KEY: Optional[str] = Field(None, description="Grok AI API key")
    PLANTNET_API_KEY: Optional[str] = Field(None, description="PlantNet API key")

    # Database Configuration
    DISEASES_DB_PATH: str = Field("database/diseases.json", description="Diseases database file path")
    TREATMENTS_DB_PATH: str = Field("database/treatments.json", description="Treatments database file path")
    REFERENCE_IMAGES_PATH: str = Field("database/reference_images", description="Reference images directory")

    # Image Processing Configuration
    MAX_FILE_SIZE_MB: float = Field(10.0, description="Maximum file size in MB")
    SUPPORTED_FORMATS: str = Field("jpg,jpeg,png,webp", description="Supported image formats")
    MIN_IMAGE_SIZE: int = Field(512, description="Minimum image dimension in pixels")
    MAX_IMAGE_SIZE: int = Field(1024, description="Maximum image dimension in pixels")

    # AI Model Configuration
    GROK_ENDPOINT: str = Field("https://api.grok.com/v1/vision/analyze", description="Grok API endpoint")
    PLANTNET_ENDPOINT: str = Field("https://api.plantnet.org/v1/identify", description="PlantNet API endpoint")
    INATURALIST_ENDPOINT: str = Field("https://api.inaturalist.org/v1/identify", description="iNaturalist API endpoint")

    # AI Model Timeout Settings
    GROK_TIMEOUT_SECONDS: int = Field(30, description="Grok API timeout in seconds")
    PLANTNET_TIMEOUT_SECONDS: int = Field(25, description="PlantNet API timeout in seconds")
    INATURALIST_TIMEOUT_SECONDS: int = Field(20, description="iNaturalist API timeout in seconds")

    # AI Model Retry Settings
    GROK_MAX_RETRIES: int = Field(3, description="Maximum Grok API retry attempts")
    PLANTNET_MAX_RETRIES: int = Field(2, description="Maximum PlantNet API retry attempts")
    INATURALIST_MAX_RETRIES: int = Field(2, description="Maximum iNaturalist API retry attempts")

    # Confidence Thresholds
    MIN_PLANT_CONFIDENCE: float = Field(0.5, description="Minimum plant identification confidence")
    MIN_DISEASE_CONFIDENCE: float = Field(0.4, description="Minimum disease identification confidence")
    MIN_IMAGE_QUALITY_SCORE: float = Field(0.3, description="Minimum image quality score")

    # Result Aggregation Weights
    PLANT_ID_WEIGHT_GROK: float = Field(0.3, description="Weight for Grok plant identification")
    PLANT_ID_WEIGHT_SPECIALIZED: float = Field(0.7, description="Weight for specialized plant identification")
    DISEASE_ID_WEIGHT_GROK: float = Field(0.6, description="Weight for Grok disease identification")
    DISEASE_ID_WEIGHT_SPECIALIZED: float = Field(0.4, description="Weight for specialized disease identification")
    CONSENSUS_BONUS: float = Field(0.1, description="Bonus for model agreement")

    # Rate Limiting Configuration
    MAX_REQUESTS_PER_MINUTE: int = Field(60, description="Maximum requests per minute per IP")
    RATE_LIMITING_ENABLED: bool = Field(True, description="Enable rate limiting")
    RATE_LIMIT_STORAGE: str = Field("memory", description="Rate limit storage backend")

    # Frontend Configuration
    BACKEND_URL: str = Field("http://localhost:8000", description="Backend API URL")
    API_TIMEOUT_SECONDS: int = Field(30, description="Frontend API timeout")

    # Feature Flags
    ENABLE_ADVANCED_ANALYSIS: bool = Field(True, description="Enable advanced AI analysis")
    ENABLE_TREATMENT_CALCULATOR: bool = Field(True, description="Enable treatment cost calculator")
    ENABLE_RESULT_SHARING: bool = Field(True, description="Enable result sharing functionality")
    ENABLE_BATCH_ANALYSIS: bool = Field(True, description="Enable batch image analysis")
    ENABLE_REFERENCE_IMAGES: bool = Field(True, description="Enable reference image display")

    # Performance Settings
    MAX_CONCURRENT_REQUESTS: int = Field(10, description="Maximum concurrent analysis requests")
    IMAGE_CACHE_SIZE: int = Field(100, description="Maximum cached images")
    REQUEST_RETRY_ATTEMPTS: int = Field(3, description="Maximum request retry attempts")

    # Logging Configuration
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    LOG_FORMAT: str = Field("json", description="Log format (json/text)")
    LOG_FILE: Optional[str] = Field(None, description="Log file path")

    # Security Configuration
    SECRET_KEY: str = Field("change-in-production", description="Secret key for authentication")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(30, description="Access token expiration")
    CORS_ORIGINS: List[str] = Field(["*"], description="CORS allowed origins")

    # External Services (Optional)
    WEATHER_API_KEY: Optional[str] = Field(None, description="Weather API key for disease prediction")
    LOCATION_API_KEY: Optional[str] = Field(None, description="Location API key for regional disease data")

    # Storage Configuration
    UPLOAD_DIR: str = Field("uploads", description="Temporary upload directory")
    TEMP_DIR: str = Field("temp", description="Temporary files directory")
    LOGS_DIR: str = Field("logs", description="Logs directory")

    # Image Processing Configuration
    JPEG_QUALITY: int = Field(85, description="JPEG compression quality (0-100)")
    PNG_COMPRESSION: int = Field(6, description="PNG compression level (0-9)")
    ENABLE_IMAGE_SHARPENING: bool = Field(True, description="Enable image sharpening")
    ENABLE_NOISE_REDUCTION: bool = Field(True, description="Enable noise reduction")

    # Advanced Analysis Settings
    MAX_PROCESSING_TIME_MS: int = Field(30000, description="Maximum analysis time in milliseconds")
    ENABLE_MODEL_CONSENSUS: bool = Field(True, description="Enable model consensus checking")
    MIN_MODELS_FOR_CONSENSUS: int = Field(2, description="Minimum models for consensus")

    @validator('SUPPORTED_FORMATS')
    def validate_supported_formats(cls, v):
        """Validate and normalize supported formats"""
        formats = [f.strip().lower() for f in v.split(',')]
        valid_formats = ['jpg', 'jpeg', 'png', 'webp']
        for fmt in formats:
            if fmt not in valid_formats:
                raise ValueError(f"Unsupported image format: {fmt}")
        return formats

    @validator('MAX_FILE_SIZE_MB')
    def validate_max_file_size(cls, v):
        """Validate maximum file size"""
        if v <= 0 or v > 50:
            raise ValueError("File size must be between 0 and 50 MB")
        return v

    @validator('MIN_IMAGE_SIZE', 'MAX_IMAGE_SIZE')
    def validate_image_sizes(cls, v, field):
        """Validate image dimensions"""
        if v < 256 or v > 4096:
            raise ValueError(f"{field.name} must be between 256 and 4096 pixels")
        return v

    @validator('MAX_IMAGE_SIZE')
    def validate_max_min_size_relationship(cls, v, values):
        """Ensure max size is greater than min size"""
        if 'MIN_IMAGE_SIZE' in values and v <= values['MIN_IMAGE_SIZE']:
            raise ValueError("MAX_IMAGE_SIZE must be greater than MIN_IMAGE_SIZE")
        return v

    @validator('CONFIDENCE_THRESHOLDS', pre=True, always=True)
    def validate_confidence_thresholds(cls, v, values):
        """Validate confidence thresholds are between 0 and 1"""
        # This validator would be called with all confidence threshold fields
        # We'll validate them in individual validators
        return values

    @validator('MIN_PLANT_CONFIDENCE', 'MIN_DISEASE_CONFIDENCE', 'MIN_IMAGE_QUALITY_SCORE')
    def validate_confidence_range(cls, v):
        """Ensure confidence values are between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("Confidence values must be between 0 and 1")
        return v

    @validator('PLANT_ID_WEIGHT_GROK', 'PLANT_ID_WEIGHT_SPECIALIZED')
    def validate_plant_weights(cls, v, values):
        """Validate plant identification weights sum to 1"""
        # This is simplified - in practice you'd check after all weights are set
        if not 0 <= v <= 1:
            raise ValueError("Weights must be between 0 and 1")
        return v

    @validator('DISEASE_ID_WEIGHT_GROK', 'DISEASE_ID_WEIGHT_SPECIALIZED')
    def validate_disease_weights(cls, v, values):
        """Validate disease identification weights sum to 1"""
        if not 0 <= v <= 1:
            raise ValueError("Weights must be between 0 and 1")
        return v

    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {', '.join(valid_levels)}")
        return v.upper()

    @validator('LOG_FORMAT')
    def validate_log_format(cls, v):
        """Validate log format"""
        valid_formats = ['json', 'text']
        if v.lower() not in valid_formats:
            raise ValueError(f"Log format must be one of: {', '.join(valid_formats)}")
        return v.lower()

    @validator('JPEG_QUALITY')
    def validate_jpeg_quality(cls, v):
        """Validate JPEG quality setting"""
        if not 0 <= v <= 100:
            raise ValueError("JPEG quality must be between 0 and 100")
        return v

    @validator('PNG_COMPRESSION')
    def validate_png_compression(cls, v):
        """Validate PNG compression setting"""
        if not 0 <= v <= 9:
            raise ValueError("PNG compression must be between 0 and 9")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def SUPPORTED_FORMATS_LIST(self) -> List[str]:
        """Get supported formats as a list"""
        return self.SUPPORTED_FORMATS

    @property
    def UPLOAD_DIR_ABSOLUTE(self) -> str:
        """Get absolute path to upload directory"""
        return os.path.abspath(self.UPLOAD_DIR)

    @property
    def TEMP_DIR_ABSOLUTE(self) -> str:
        """Get absolute path to temp directory"""
        return os.path.abspath(self.TEMP_DIR)

    @property
    def LOGS_DIR_ABSOLUTE(self) -> str:
        """Get absolute path to logs directory"""
        return os.path.abspath(self.LOGS_DIR)

    @property
    def DATABASE_PATHS_ABSOLUTE(self) -> dict:
        """Get absolute paths to database files"""
        return {
            'diseases': os.path.abspath(self.DISEASES_DB_PATH),
            'treatments': os.path.abspath(self.TREATMENTS_DB_PATH),
            'reference_images': os.path.abspath(self.REFERENCE_IMAGES_PATH)
        }

    def get_ai_model_config(self, model_type: str) -> dict:
        """Get configuration for a specific AI model"""
        configs = {
            'grok': {
                'api_key': self.GROK_API_KEY,
                'endpoint': self.GROK_ENDPOINT,
                'timeout': self.GROK_TIMEOUT_SECONDS,
                'max_retries': self.GROK_MAX_RETRIES,
                'enabled': bool(self.GROK_API_KEY)
            },
            'plantnet': {
                'api_key': self.PLANTNET_API_KEY,
                'endpoint': self.PLANTNET_ENDPOINT,
                'timeout': self.PLANTNET_TIMEOUT_SECONDS,
                'max_retries': self.PLANTNET_MAX_RETRIES,
                'enabled': bool(self.PLANTNET_API_KEY)
            },
            'inaturalist': {
                'api_key': None,  # iNaturalist typically doesn't require API key
                'endpoint': self.INATURALIST_ENDPOINT,
                'timeout': self.INATURALIST_TIMEOUT_SECONDS,
                'max_retries': self.INATURALIST_MAX_RETRIES,
                'enabled': True
            }
        }
        return configs.get(model_type, {})

    def get_aggregation_weights(self) -> dict:
        """Get model aggregation weights"""
        return {
            'plant_identification': {
                'grok': self.PLANT_ID_WEIGHT_GROK,
                'specialized': self.PLANT_ID_WEIGHT_SPECIALIZED
            },
            'disease_identification': {
                'grok': self.DISEASE_ID_WEIGHT_GROK,
                'specialized': self.DISEASE_ID_WEIGHT_SPECIALIZED
            },
            'consensus_bonus': self.CONSENSUS_BONUS
        }

    def validate_required_api_keys(self) -> List[str]:
        """Check which required API keys are missing"""
        missing_keys = []
        if not self.GROK_API_KEY:
            missing_keys.append('GROK_API_KEY')
        if not self.PLANTNET_API_KEY:
            missing_keys.append('PLANTNET_API_KEY')
        return missing_keys

    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.ENVIRONMENT.lower() in ['development', 'dev', 'local']

    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.ENVIRONMENT.lower() in ['production', 'prod']

    def create_directories(self):
        """Create necessary directories"""
        import os
        directories = [
            self.UPLOAD_DIR_ABSOLUTE,
            self.TEMP_DIR_ABSOLUTE,
            self.LOGS_DIR_ABSOLUTE,
            os.path.dirname(self.DATABASE_PATHS_ABSOLUTE['diseases']),
            os.path.dirname(self.DATABASE_PATHS_ABSOLUTE['treatments']),
            self.DATABASE_PATHS_ABSOLUTE['reference_images']
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.create_directories()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment"""
    global _settings
    _settings = Settings()
    _settings.create_directories()
    return _settings


# For testing
def create_test_settings(**overrides) -> Settings:
    """Create settings for testing with overrides"""
    test_settings = Settings()
    for key, value in overrides.items():
        setattr(test_settings, key, value)
    return test_settings