"""
FastAPI Routes

This module defines all API endpoints for image analysis, disease information,
reference images, and health monitoring.
"""

import asyncio
import logging
import time
import uuid
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import io

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer
from fastapi.limiter import Limiter
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request as StarletteRequest
import httpx

from backend.models.disease import (
    Disease, TreatmentInfo, DiseaseInfoResponse, DatabaseStats
)
from backend.models.analysis import (
    ImageAnalysisResult, AnalysisRequest, AnalysisResponse,
    AIModelType, ProcessingStatus, ProcessingError,
    ConfidenceAggregation, ImageProcessingResponse
)
from backend.services.config import Settings, get_settings
from backend.services.image_processor import ImageProcessor
from backend.services.ai_service import AIService
from backend.services.disease_service import DiseaseService

logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=lambda request: request.client.host)
security = HTTPBearer(auto_error=False)

# Create router
router = APIRouter()

# Background task for cleanup
async def cleanup_temp_files(image_path: str):
    """Clean up temporary files after analysis"""
    try:
        Path(image_path).unlink(missing_ok=True)
        logger.debug(f"Cleaned up temporary file: {image_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary file {image_path}: {str(e)}")


def get_ai_service(request: Request) -> AIService:
    """Get AI service from app state"""
    ai_service = getattr(request.app.state, 'ai_service', None)
    if not ai_service:
        raise HTTPException(status_code=503, detail="AI service not available")
    return ai_service


def get_image_processor(request: Request) -> ImageProcessor:
    """Get image processor from app state"""
    image_processor = getattr(request.app.state, 'image_processor', None)
    if not image_processor:
        raise HTTPException(status_code=503, detail="Image processor not available")
    return image_processor


def get_disease_service(request: Request) -> DiseaseService:
    """Get disease service from app state"""
    disease_service = getattr(request.app.state, 'disease_service', None)
    if not disease_service:
        raise HTTPException(status_code=503, detail="Disease service not available")
    return disease_service


def validate_image_file(file: UploadFile, settings: Settings) -> None:
    """Validate uploaded image file"""
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Supported formats: {settings.SUPPORTED_FORMATS_LIST}"
        )

    if file.size and file.size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB"
        )


# Main analysis endpoint
@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyze plant leaf image",
    description="Upload a plant leaf image for AI-powered disease detection and plant identification",
    tags=["Analysis"]
)
@limiter.limit("60/minute")
async def analyze_image(
    request: StarletteRequest,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Plant leaf image to analyze"),
    quality_threshold: Optional[float] = None,
    include_alternatives: Optional[bool] = True,
    max_processing_time_ms: Optional[int] = None,
    settings: Settings = Depends(get_settings),
    ai_service: AIService = Depends(get_ai_service),
    image_processor: ImageProcessor = Depends(get_image_processor)
):
    """Analyze plant leaf image for disease detection"""
    session_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Get optional parameters from settings if not provided
        quality_threshold = quality_threshold or settings.MIN_IMAGE_QUALITY_SCORE
        max_processing_time_ms = max_processing_time_ms or settings.MAX_PROCESSING_TIME_MS

        # Validate image file
        validate_image_file(image, settings)

        # Read image data
        image_data = await image.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty image file")

        logger.info(
            f"analysis_started",
            session_id=session_id,
            filename=image.filename,
            file_size=len(image_data),
            content_type=image.content_type
        )

        # Process image
        image_analysis = await image_processor.analyze_image(image_data)

        if not image_analysis.is_acceptable:
            processing_time_ms = int((time.time() - start_time) * 1000)

            return AnalysisResponse(
                success=False,
                session_id=session_id,
                timestamp=time.time(),
                results=None,
                error_message=f"Image quality too low ({image_analysis.quality_metrics.overall_score:.2f}). {', '.join(image_analysis.processing_recommendations)}",
                processing_time_ms=processing_time_ms
            )

        # Analyze with AI models
        model_results = await ai_service.analyze_with_all_models(image_data)

        if not model_results:
            raise HTTPException(status_code=503, detail="All AI models failed to process image")

        # Aggregate results
        aggregated_results = ai_service.aggregate_results(model_results)

        # Build complete results
        analysis_results = {
            "plant_detection": {
                "plant_name": aggregated_results.primary_plant_name,
                "confidence": aggregated_results.plant_confidence,
                "alternative_plants": aggregated_results.plant_alternatives
            },
            "disease_detection": {
                "primary_disease": {
                    "disease_name": aggregated_results.primary_disease_name,
                    "confidence": aggregated_results.disease_confidence
                } if aggregated_results.primary_disease_name else None,
                "secondary_diseases": aggregated_results.disease_alternatives
            },
            "ai_model_results": {
                model.model_type.value: {
                    "success": model.success,
                    "plant_confidence": model.plant_identification.confidence if model.plant_identification else 0.0,
                    "disease_confidence": model.disease_identification.confidence if model.disease_identification else 0.0,
                    "processing_time_ms": model.processing_time_ms
                } for model in model_results
            },
            "metadata": {
                "image_quality_score": image_analysis.quality_metrics.overall_score,
                "processing_time_total_ms": int((time.time() - start_time) * 1000),
                "models_used": [model.model_type.value for model in model_results if model.success]
            }
        }

        # Schedule cleanup of processed image
        if hasattr(image_analysis, 'processed_image_path'):
            background_tasks.add_task(cleanup_temp_files, image_analysis.processed_image_path)

        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"analysis_completed",
            session_id=session_id,
            success=True,
            processing_time_ms=processing_time_ms,
            primary_plant=aggregated_results.primary_plant_name,
            primary_disease=aggregated_results.primary_disease_name
        )

        return AnalysisResponse(
            success=True,
            session_id=session_id,
            timestamp=time.time(),
            results=analysis_results,
            processing_time_ms=processing_time_ms,
            warnings=image_analysis.processing_recommendations if image_analysis.quality_metrics.overall_score < 0.7 else []
        )

    except HTTPException:
        raise
    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.error(
            f"analysis_failed",
            session_id=session_id,
            error=str(e),
            processing_time_ms=processing_time_ms
        )

        return AnalysisResponse(
            success=False,
            session_id=session_id,
            timestamp=time.time(),
            results=None,
            error_message=f"Analysis failed: {str(e)}",
            processing_time_ms=processing_time_ms
        )


# Disease information endpoint
@router.get(
    "/diseases/{disease_id}",
    response_model=DiseaseInfoResponse,
    summary="Get disease information",
    description="Get detailed information about a specific disease including symptoms, causes, and treatments",
    tags=["Diseases"]
)
@limiter.limit("120/minute")
async def get_disease_info(
    disease_id: str,
    request: StarletteRequest,
    include_treatments: Optional[bool] = True,
    include_images: Optional[bool] = True,
    disease_service: DiseaseService = Depends(get_disease_service)
):
    """Get comprehensive disease information"""
    try:
        # Get disease information
        disease = await disease_service.get_disease_by_id(disease_id)
        if not disease:
            raise HTTPException(status_code=404, detail=f"Disease '{disease_id}' not found")

        # Get treatment information if requested
        treatments = None
        if include_treatments:
            treatments = await disease_service.get_treatment_recommendations(disease_id)

        # Get reference images if requested
        if include_images:
            reference_images = await disease_service.get_reference_images(disease_id)
        else:
            reference_images = []

        # Update disease with image paths
        disease.reference_images = reference_images

        logger.info(f"disease_info_requested", disease_id=disease_id, include_treatments=include_treatments)

        return DiseaseInfoResponse(
            success=True,
            disease=disease,
            treatments=treatments
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get disease info for {disease_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve disease information")


# Reference images endpoint
@router.get(
    "/images/reference/{disease_id}",
    summary="Get reference disease images",
    description="Get reference images showing disease symptoms for visual comparison",
    tags=["Images"]
)
@limiter.limit("60/minute")
async def get_reference_images(
    disease_id: str,
    request: StarletteRequest,
    image_index: Optional[int] = 0,
    disease_service: DiseaseService = Depends(get_disease_service)
):
    """Get reference image for a disease"""
    try:
        # Get reference images
        reference_images = await disease_service.get_reference_images(disease_id)
        if not reference_images:
            raise HTTPException(status_code=404, detail=f"No reference images found for disease '{disease_id}'")

        # Validate image index
        if image_index >= len(reference_images):
            raise HTTPException(status_code=400, detail=f"Image index {image_index} out of range. Available images: 0-{len(reference_images)-1}")

        image_path = reference_images[image_index]
        full_path = Path(image_path)

        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"Reference image file not found: {image_path}")

        # Determine content type
        if image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
            media_type = 'image/jpeg'
        elif image_path.lower().endswith('.png'):
            media_type = 'image/png'
        elif image_path.lower().endswith('.webp'):
            media_type = 'image/webp'
        else:
            media_type = 'image/jpeg'  # Default

        logger.info(f"reference_image_served", disease_id=disease_id, image_index=image_index, image_path=image_path)

        return FileResponse(
            path=str(full_path),
            media_type=media_type,
            filename=f"{disease_id}_reference_{image_index}.jpg"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve reference image for {disease_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to serve reference image")


# Search endpoint
@router.get(
    "/search/diseases",
    summary="Search diseases",
    description="Search diseases by name, symptoms, or causes",
    tags=["Search"]
)
@limiter.limit("120/minute")
async def search_diseases(
    q: str = None,  # Search query
    limit: Optional[int] = 10,
    plant: Optional[str] = None,
    disease_service: DiseaseService = Depends(get_disease_service)
):
    """Search diseases in the database"""
    try:
        if not q and not plant:
            raise HTTPException(status_code=400, detail="Search query 'q' or 'plant' parameter is required")

        if limit <= 0 or limit > 50:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 50")

        results = []

        if plant:
            # Search by plant name
            diseases = await disease_service.get_diseases_by_plant(plant)
            results.extend(diseases)
        elif q:
            # General search
            diseases = await disease_service.search_diseases(q, limit)
            results.extend(diseases)

        # Format results
        search_results = [
            {
                "disease_id": disease.disease_id,
                "plant_name": disease.plant_name,
                "disease_name": disease.disease_name,
                "scientific_name": disease.scientific_name,
                "description": disease.description[:200] + "..." if len(disease.description) > 200 else disease.description
            }
            for disease in results[:limit]
        ]

        logger.info(f"disease_search", query=q, plant=plant, results_count=len(search_results))

        return {
            "success": True,
            "results": search_results,
            "total_count": len(search_results),
            "query": q,
            "plant": plant
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Disease search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Search failed")


# Get supported plants endpoint
@router.get(
    "/plants/supported",
    summary="Get supported plants",
    description="Get list of all plants supported by the disease detection system",
    tags=["Plants"]
)
@limiter.limit("60/minute")
async def get_supported_plants(
    disease_service: DiseaseService = Depends(get_disease_service)
):
    """Get list of supported plants"""
    try:
        plants = await disease_service.get_supported_plants()

        return {
            "success": True,
            "plants": plants,
            "total_count": len(plants)
        }

    except Exception as e:
        logger.error(f"Failed to get supported plants: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve supported plants")


# Get disease categories endpoint
@router.get(
    "/diseases/categories",
    summary="Get disease categories",
    description="Get diseases categorized by crop type (vegetables, fruits, grains, legumes, other)",
    tags=["Diseases"]
)
@limiter.limit("60/minute")
async def get_disease_categories(
    disease_service: DiseaseService = Depends(get_disease_service)
):
    """Get diseases categorized by crop type"""
    try:
        categories = await disease_service.get_disease_categories()

        return {
            "success": True,
            "categories": categories,
            "total_categories": len(categories),
            "total_diseases": sum(len(diseases) for diseases in categories.values())
        }

    except Exception as e:
        logger.error(f"Failed to get disease categories: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve disease categories")


# AI models status endpoint
@router.get(
    "/models/status",
    summary="Get AI models status",
    description="Get status and health of all AI models",
    tags=["Models"]
)
@limiter.limit("60/minute")
async def get_models_status(
    request: StarletteRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """Get status of all AI models"""
    try:
        available_models = await ai_service.get_available_models()
        model_stats = ai_service.get_model_statistics()

        # Check health of each model
        health_status = {}
        for model_type in available_models:
            if model_type == "grok":
                health_status[model_type] = await ai_service.check_grok_health()
            elif model_type == "plantnet":
                health_status[model_type] = await ai_service.check_plantnet_health()
            elif model_type == "inaturalist":
                health_status[model_type] = await ai_service.check_inaturalist_health()

        return {
            "success": True,
            "available_models": available_models,
            "health_status": health_status,
            "statistics": model_stats,
            "total_models": len(available_models)
        }

    except Exception as e:
        logger.error(f"Failed to get models status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models status")


# Batch analysis endpoint
@router.post(
    "/analyze/batch",
    summary="Batch analyze images",
    description="Analyze multiple plant leaf images in a single request",
    tags=["Analysis"]
)
@limiter.limit("10/minute")
async def batch_analyze_images(
    request: StarletteRequest,
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(..., description="Multiple plant leaf images to analyze"),
    max_concurrent: Optional[int] = 3,
    settings: Settings = Depends(get_settings),
    ai_service: AIService = Depends(get_ai_service),
    image_processor: ImageProcessor = Depends(get_image_processor)
):
    """Analyze multiple images in batch"""
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed per batch")

    if len(images) < 2:
        raise HTTPException(status_code=400, detail="At least 2 images required for batch analysis")

    batch_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Validate all images
        for image in images:
            validate_image_file(image, settings)

        # Read all images
        image_data_list = []
        for image in images:
            data = await image.read()
            if not data:
                raise HTTPException(status_code=400, detail=f"Empty image file: {image.filename}")
            image_data_list.append((image.filename, data))

        logger.info(
            f"batch_analysis_started",
            batch_id=batch_id,
            image_count=len(images),
            total_size=sum(len(data) for _, data in image_data_list)
        )

        # Process images concurrently
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single_image(filename: str, data: bytes):
            async with semaphore:
                try:
                    # Process image
                    image_analysis = await image_processor.analyze_image(data)

                    if not image_analysis.is_acceptable:
                        return {
                            "success": False,
                            "filename": filename,
                            "error": f"Image quality too low: {', '.join(image_analysis.processing_recommendations)}"
                        }

                    # Analyze with AI
                    model_results = await ai_service.analyze_with_all_models(data)
                    if not model_results:
                        return {
                            "success": False,
                            "filename": filename,
                            "error": "All AI models failed"
                        }

                    # Aggregate results
                    aggregated_results = ai_service.aggregate_results(model_results)

                    return {
                        "success": True,
                        "filename": filename,
                        "results": {
                            "plant_detection": {
                                "plant_name": aggregated_results.primary_plant_name,
                                "confidence": aggregated_results.plant_confidence,
                                "alternative_plants": aggregated_results.plant_alternatives
                            },
                            "disease_detection": {
                                "primary_disease": {
                                    "disease_name": aggregated_results.primary_disease_name,
                                    "confidence": aggregated_results.disease_confidence
                                } if aggregated_results.primary_disease_name else None,
                                "secondary_diseases": aggregated_results.disease_alternatives
                            },
                            "metadata": {
                                "image_quality_score": image_analysis.quality_metrics.overall_score
                            }
                        }
                    }

                except Exception as e:
                    return {
                        "success": False,
                        "filename": filename,
                        "error": str(e)
                    }

        # Process all images
        tasks = [process_single_image(filename, data) for filename, data in image_data_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes and failures
        successful_analyses = sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
        failed_analyses = len(results) - successful_analyses

        total_processing_time_ms = int((time.time() - start_time) * 1000)
        average_processing_time_ms = total_processing_time_ms // len(images)

        logger.info(
            f"batch_analysis_completed",
            batch_id=batch_id,
            successful=successful_analyses,
            failed=failed_analyses,
            total_processing_time_ms=total_processing_time_ms
        )

        return {
            "success": True,
            "batch_id": batch_id,
            "total_images": len(images),
            "successful_analyses": successful_analyses,
            "failed_analyses": failed_analyses,
            "total_processing_time_ms": total_processing_time_ms,
            "average_processing_time_ms": average_processing_time_ms,
            "individual_results": [r for r in results if isinstance(r, dict)]
        }

    except HTTPException:
        raise
    except Exception as e:
        total_processing_time_ms = int((time.time() - start_time) * 1000)

        logger.error(
            f"batch_analysis_failed",
            batch_id=batch_id,
            error=str(e),
            total_processing_time_ms=total_processing_time_ms
        )

        return {
            "success": False,
            "batch_id": batch_id,
            "total_images": len(images),
            "successful_analyses": 0,
            "failed_analyses": len(images),
            "total_processing_time_ms": total_processing_time_ms,
            "average_processing_time_ms": 0,
            "individual_results": [],
            "error": f"Batch analysis failed: {str(e)}"
        }


# Export endpoint for disease data
@router.get(
    "/export/diseases",
    summary="Export disease database",
    description="Export the complete disease database as JSON",
    tags=["Export"]
)
@limiter.limit("5/minute")
async def export_diseases(
    disease_service: DiseaseService = Depends(get_disease_service)
):
    """Export disease database as JSON"""
    try:
        # Get all diseases
        diseases = list(disease_service.diseases_data.values())

        export_data = {
            "export_timestamp": time.time(),
            "total_diseases": len(diseases),
            "diseases": [disease.dict() for disease in diseases]
        }

        return JSONResponse(
            content=export_data,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=diseases_export.json"}
        )

    except Exception as e:
        logger.error(f"Failed to export diseases: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export disease database")


# Statistics endpoint
@router.get(
    "/stats",
    summary="Get API statistics",
    description="Get comprehensive statistics about the disease database and API usage",
    tags=["Statistics"]
)
@limiter.limit("60/minute")
async def get_api_statistics(
    request: StarletteRequest,
    disease_service: DiseaseService = Depends(get_disease_service),
    ai_service: AIService = Depends(get_ai_service)
):
    """Get comprehensive API statistics"""
    try:
        # Get database stats
        db_stats = await disease_service.get_database_stats()

        # Get AI model stats
        model_stats = ai_service.get_model_statistics()

        # Get available models
        available_models = await ai_service.get_available_models()

        statistics = {
            "database": db_stats.dict(),
            "ai_models": {
                **model_stats,
                "available_count": len(available_models),
                "available_models": available_models
            },
            "api_info": {
                "version": "1.0.0",
                "endpoints_count": 10,
                "supported_formats": ["jpg", "jpeg", "png", "webp"],
                "max_file_size_mb": disease_service.settings.MAX_FILE_SIZE_MB,
                "max_concurrent_requests": disease_service.settings.MAX_CONCURRENT_REQUESTS
            }
        }

        return {
            "success": True,
            "statistics": statistics,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Failed to get API statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")