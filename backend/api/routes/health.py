"""
Health check endpoints for the Plant Disease Detection API
"""

import logging
import os
import psutil
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", summary="Health Check")
async def health_check():
    """
    Basic health check endpoint to verify API is running
    """
    try:
        return {
            "success": True,
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "Plant Disease Detection API",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/health/detailed", summary="Detailed Health Check")
async def detailed_health_check():
    """
    Detailed health check with system information
    """
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Model information (placeholder - would check actual model availability)
        model_path = os.getenv("MODEL_PATH", "./backend/ml_models/disease_classifier.pkl")
        model_available = os.path.exists(model_path) if model_path else False

        # Database information
        db_path = os.getenv("DATABASE_URL", "./data/disease_database.json")
        db_available = os.path.exists(db_path) if db_path else False

        health_data = {
            "success": True,
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "Plant Disease Detection API",
            "version": "1.0.0",
            "system": {
                "cpu_percent": f"{cpu_percent:.1f}%",
                "memory": {
                    "total": f"{memory.total / (1024**3):.1f}GB",
                    "available": f"{memory.available / (1024**3):.1f}GB",
                    "percent_used": f"{memory.percent:.1f}%"
                },
                "disk": {
                    "total": f"{disk.total / (1024**3):.1f}GB",
                    "free": f"{disk.free / (1024**3):.1f}GB",
                    "percent_used": f"{disk.percent:.1f}%"
                }
            },
            "services": {
                "model_loaded": model_available,
                "database_available": db_available,
                "model_path": model_path,
                "database_path": db_path
            }
        }

        # Determine overall health
        if not model_available or not db_available:
            health_data["status"] = "degraded"

        return health_data

    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Health check service unavailable")


@router.get("/health/ready", summary="Readiness Check")
async def readiness_check():
    """
    Kubernetes-style readiness check - verifies service is ready to accept requests
    """
    try:
        model_path = os.getenv("MODEL_PATH", "./backend/ml_models/disease_classifier.pkl")
        db_path = os.getenv("DATABASE_URL", "./data/disease_database.json")

        model_ready = os.path.exists(model_path) if model_path else False
        db_ready = os.path.exists(db_path) if db_path else False

        if model_ready and db_ready:
            return {
                "success": True,
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat(),
                "checks": {
                    "model": "ready",
                    "database": "ready"
                }
            }
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "status": "not ready",
                    "timestamp": datetime.utcnow().isoformat(),
                    "checks": {
                        "model": "ready" if model_ready else "not ready",
                        "database": "ready" if db_ready else "not ready"
                    }
                }
            )

    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "status": "not ready",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )