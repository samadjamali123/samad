"""
Plant Disease Detection API - Main FastAPI Application
"""

import os
import logging
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn

from api.routes import detection, diseases, health
from api.middleware import setup_logging, setup_exception_handlers

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Plant Disease Detection API...")
    yield
    # Shutdown
    logger.info("Shutting down Plant Disease Detection API...")


# Create FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    version="1.0.0",
    description="AI-powered plant leaf disease detection API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS middleware
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(
    detection.router,
    prefix="/api",
    tags=["detection"],
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid image format or size"},
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"description": "File too large"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Model inference failure"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Detection service unavailable"}
    }
)

app.include_router(
    diseases.router,
    prefix="/api",
    tags=["diseases"],
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Disease not found"}
    }
)

app.include_router(
    health.router,
    prefix="/api",
    tags=["health"]
)

# Serve static files (for disease images)
app.mount("/static", StaticFiles(directory="data/static"), name="static")

# Setup exception handlers
setup_exception_handlers(app)


@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Plant Disease Detection API",
        "version": "1.0.0",
        "description": "AI-powered plant leaf disease detection system",
        "docs_url": "/docs",
        "health_check": "/api/health"
    }


if __name__ == "__main__":
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    host = os.getenv("API_HOST", "localhost")
    port = int(os.getenv("API_PORT", 8000))

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug_mode,
        log_level="info" if not debug_mode else "debug"
    )