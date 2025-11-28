"""
Plant Leaf Disease Detector - FastAPI Main Application

This is the main entry point for the plant leaf disease detection backend API.
It provides endpoints for image analysis, disease information, and health monitoring.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import application components
from backend.models.disease import HealthCheckResponse, DatabaseStats
from backend.models.analysis import ProcessingStatus
from backend.services.config import Settings, get_settings
from backend.services.disease_service import DiseaseService
from backend.services.image_processor import ImageProcessor
from backend.services.ai_service import AIService
from backend.api.routes import router as api_router
from backend.utils.logging import setup_logging

# Global services
disease_service: Optional[DiseaseService] = None
image_processor: Optional[ImageProcessor] = None
ai_service: Optional[AIService] = None
settings: Optional[Settings] = None
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    global disease_service, image_processor, ai_service, settings

    
    logger.info("Starting Plant Leaf Disease Detector API...")

    try:
        # Load settings
        settings = get_settings()
        logger.info(f"Loaded configuration for environment: {settings.ENVIRONMENT}")

        # Initialize services
        disease_service = DiseaseService()
        await disease_service.initialize()
        logger.info("Disease service initialized")

        image_processor = ImageProcessor(settings)
        logger.info("Image processor initialized")

        ai_service = AIService(settings)
        await ai_service.initialize()
        logger.info("AI service initialized")

        # Store services in app state
        app.state.disease_service = disease_service
        app.state.image_processor = image_processor
        app.state.ai_service = ai_service
        app.state.settings = settings

        logger.info("All services initialized successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise

    # Shutdown
    logger.info("Shutting down Plant Leaf Disease Detector API...")

    if disease_service:
        await disease_service.cleanup()
    if ai_service:
        await ai_service.cleanup()

    logger.info("Shutdown complete")


app = FastAPI(
    title="Plant Leaf Disease Detector",
    description="AI-powered plant disease detection and treatment recommendation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Setup logging
setup_logging()

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests"""
    start_time = time.time()

    # Generate request ID
    import uuid
    request_id = str(uuid.uuid4())

    # Log request
    logger.info(
        "request_started",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None
    )

    # Process request
    response = await call_next(request)

    # Calculate processing time
    processing_time = int((time.time() - start_time) * 1000)

    # Log response
    logger.info(
        "request_completed",
        request_id=request_id,
        status_code=response.status_code,
        processing_time_ms=processing_time
    )

    # Add custom headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Processing-Time"] = str(processing_time)

    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(
        "unexpected_error",
        request_id=request.headers.get("X-Request-ID"),
        error=str(exc),
        url=str(request.url)
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic API information"""
    return {
        "name": "Plant Leaf Disease Detector API",
        "version": "1.0.0",
        "description": "AI-powered plant disease detection and treatment recommendation",
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
        "timestamp": time.time()
    }


# Health check endpoint
@app.get("/api/v1/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Check API and external service health"""
    try:
        # Check services
        services_status = {}

        # Check AI services
        if app.state.ai_service:
            services_status["grok_api"] = await app.state.ai_service.check_grok_health()
            services_status["plantnet_api"] = await app.state.ai_service.check_plantnet_health()
        else:
            services_status["ai_service"] = False

        # Check database services
        if app.state.disease_service:
            services_status["disease_database"] = await app.state.disease_service.health_check()
        else:
            services_status["disease_service"] = False

        # Overall status
        overall_status = "healthy" if all(services_status.values()) else "degraded"

        # Get database stats
        stats = {}
        if app.state.disease_service:
            stats = await app.state.disease_service.get_database_stats()

        return HealthCheckResponse(
            status=overall_status,
            timestamp=time.time(),
            services=services_status,
            version="1.0.0",
            uptime_seconds=int(time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0)
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=time.time(),
            services={},
            version="1.0.0",
            uptime_seconds=0
        )


# Database stats endpoint
@app.get("/api/v1/stats", response_model=DatabaseStats, tags=["Database"])
async def get_database_stats():
    """Get database statistics"""
    if not app.state.disease_service:
        raise HTTPException(status_code=503, detail="Disease service not available")

    try:
        stats = await app.state.disease_service.get_database_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get database stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve database statistics")


# Configuration endpoint
@app.get("/api/v1/config", tags=["Configuration"])
async def get_configuration():
    """Get public configuration information"""
    if not app.state.settings:
        raise HTTPException(status_code=503, detail="Settings not available")

    return {
        "max_file_size_mb": app.state.settings.MAX_FILE_SIZE_MB,
        "supported_formats": app.state.settings.SUPPORTED_FORMATS,
        "min_image_size": app.state.settings.MIN_IMAGE_SIZE,
        "max_requests_per_minute": app.state.settings.MAX_REQUESTS_PER_MINUTE,
        "enabled_features": {
            "advanced_analysis": app.state.settings.ENABLE_ADVANCED_ANALYSIS,
            "treatment_calculator": app.state.settings.ENABLE_TREATMENT_CALCULATOR,
            "result_sharing": app.state.settings.ENABLE_RESULT_SHARING
        },
        "ai_models": await app.state.ai_service.get_available_models() if app.state.ai_service else []
    }


# Include API routes
app.include_router(api_router, prefix="/api/v1")


# Startup event (for compatibility with FastAPI <0.21.0)
@app.on_event("startup")
async def startup_event():
    """Store startup time"""
    app.state.start_time = time.time()


# Shutdown event (for compatibility with FastAPI <0.21.0)
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if hasattr(app.state, 'disease_service') and app.state.disease_service:
        await app.state.disease_service.cleanup()
    if hasattr(app.state, 'ai_service') and app.state.ai_service:
        await app.state.ai_service.cleanup()


def create_app() -> FastAPI:
    """Factory function to create FastAPI application"""
    return app


if __name__ == "__main__":
    # Run development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
