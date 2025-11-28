"""
Disease detection endpoints for the Plant Disease Detection API
"""

import logging
import os
import io
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()

# Import image processing and detection services
from ..services.image_processor import ImageProcessor
from ..services.disease_detector import DiseaseDetector

# Initialize services
try:
    image_processor = ImageProcessor()
    disease_detector = DiseaseDetector()
    logger.info("Detection services initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize detection services: {str(e)}")
    image_processor = None
    disease_detector = None


# Pydantic models for detection responses
class DetectionPrediction(BaseModel):
    disease_name: str
    scientific_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    class_index: int
    severity: str
    detected_at: str


class ImageInfo(BaseModel):
    filename: str
    size: int
    dimensions: List[int] = Field(min_items=2, max_items=2)
    format: str


class DetectionResponse(BaseModel):
    success: bool
    prediction: DetectionPrediction
    image_info: ImageInfo
    processing_time: float = Field(description="Processing time in seconds")


class DetectionStatusResponse(BaseModel):
    success: bool
    service_available: bool
    model_loaded: bool
    supported_formats: List[str]
    max_file_size: int
    confidence_threshold: float


@router.post("/detect", response_model=DetectionResponse, summary="Detect Plant Disease")
async def detect_disease(
    image: UploadFile = File(..., description="Plant leaf image for disease detection"),
    confidence_threshold: float = Form(0.6, ge=0.0, le=1.0, description="Minimum confidence threshold for predictions")
):
    """
    Main disease detection endpoint that analyzes uploaded plant leaf images
    and returns disease predictions with confidence scores.

    Supported formats: JPEG, PNG, WebP
    Maximum file size: 10MB
    """
    start_time = datetime.now()

    # Check if services are available
    if not image_processor or not disease_detector:
        raise HTTPException(
            status_code=503,
            detail="Detection service is currently unavailable"
        )

    # Validate file format
    allowed_extensions = os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png,webp").split(",")
    file_extension = image.filename.lower().split('.')[-1] if '.' in image.filename else ""

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )

    # Check file size
    max_size = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB default
    file_content = await image.read()

    if len(file_content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB"
        )

    if len(file_content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded"
        )

    try:
        # Process image and detect disease
        # Reset file pointer for PIL processing
        image.file.seek(0)

        # Open and validate image
        try:
            pil_image = Image.open(image.file)

            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            image_dimensions = list(pil_image.size)
            image_format = pil_image.format or file_extension.upper()

        except Exception as e:
            logger.error(f"Invalid image file: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="Invalid image file or corrupted image data"
            )

        # Preprocess image
        try:
            processed_image = image_processor.preprocess_image(pil_image)
            logger.info(f"Image preprocessed successfully. Shape: {processed_image.shape}")
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Image preprocessing failed"
            )

        # Run disease detection
        try:
            prediction = disease_detector.predict(processed_image, confidence_threshold)
            logger.info(f"Disease detection completed. Prediction: {prediction}")

            if not prediction:
                raise HTTPException(
                    status_code=500,
                    detail="No valid prediction could be made"
                )

        except Exception as e:
            logger.error(f"Disease detection failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Disease detection model inference failed"
            )

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        # Create response
        detection_response = DetectionResponse(
            success=True,
            prediction=DetectionPrediction(
                disease_name=prediction['disease_name'],
                scientific_name=prediction['scientific_name'],
                confidence=prediction['confidence'],
                class_index=prediction['class_index'],
                severity=prediction['severity'],
                detected_at=datetime.utcnow().isoformat()
            ),
            image_info=ImageInfo(
                filename=image.filename,
                size=len(file_content),
                dimensions=image_dimensions,
                format=image_format
            ),
            processing_time=processing_time
        )

        logger.info(f"Detection completed successfully in {processing_time:.2f}s")
        return detection_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during detection: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during disease detection"
        )


@router.post("/detect/batch", summary="Batch Disease Detection")
async def detect_diseases_batch(
    images: List[UploadFile] = File(..., description="Multiple plant leaf images for batch disease detection"),
    confidence_threshold: float = Form(0.6, ge=0.0, le=1.0, description="Minimum confidence threshold for predictions")
):
    """
    Batch disease detection endpoint for processing multiple images at once.
    Maximum 10 images per batch request.

    Returns individual results for each image along with batch statistics.
    """
    if not image_processor or not disease_detector:
        raise HTTPException(
            status_code=503,
            detail="Detection service is currently unavailable"
        )

    # Validate batch size
    max_batch_size = 10
    if len(images) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Too many images in batch. Maximum allowed: {max_batch_size}"
        )

    if len(images) == 0:
        raise HTTPException(
            status_code=400,
            detail="No images provided in batch"
        )

    results = []
    start_time = datetime.now()
    successful_detections = 0
    failed_detections = 0

    for i, image in enumerate(images):
        try:
            # Reset file pointer
            image.file.seek(0)

            # Process individual image
            individual_result = await detect_disease(image, confidence_threshold)
            individual_result.image_info.filename = f"image_{i+1}_{individual_result.image_info.filename}"
            results.append(individual_result.dict())
            successful_detections += 1

        except HTTPException as e:
            failed_detections += 1
            results.append({
                "success": False,
                "error": e.detail,
                "filename": f"image_{i+1}_{image.filename}",
                "processing_time": 0.0
            })
        except Exception as e:
            failed_detections += 1
            logger.error(f"Unexpected error processing image {i+1}: {str(e)}")
            results.append({
                "success": False,
                "error": "Unexpected processing error",
                "filename": f"image_{i+1}_{image.filename}",
                "processing_time": 0.0
            })

    # Calculate batch statistics
    total_processing_time = (datetime.now() - start_time).total_seconds()

    batch_response = {
        "success": successful_detections > 0,
        "batch_info": {
            "total_images": len(images),
            "successful_detections": successful_detections,
            "failed_detections": failed_detections,
            "success_rate": successful_detections / len(images),
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(images)
        },
        "results": results
    }

    return batch_response


@router.get("/detect/status", response_model=DetectionStatusResponse, summary="Detection Service Status")
async def get_detection_status():
    """
    Check the status of the disease detection service
    """
    try:
        service_available = image_processor is not None and disease_detector is not None
        model_loaded = disease_detector.model_loaded if disease_detector else False

        supported_formats = os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png,webp").split(",")
        max_file_size = int(os.getenv("MAX_FILE_SIZE", "10485760"))
        confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

        return DetectionStatusResponse(
            success=True,
            service_available=service_available,
            model_loaded=model_loaded,
            supported_formats=supported_formats,
            max_file_size=max_file_size,
            confidence_threshold=confidence_threshold
        )

    except Exception as e:
        logger.error(f"Error getting detection status: {str(e)}")
        return DetectionStatusResponse(
            success=False,
            service_available=False,
            model_loaded=False,
            supported_formats=[],
            max_file_size=0,
            confidence_threshold=0.0
        )