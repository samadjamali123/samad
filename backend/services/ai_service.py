"""
AI Service Integration

This module handles integration with multiple AI services including Grok, PlantNet,
iNaturalist, and other specialized plant disease detection models.
"""

import asyncio
import io
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import base64
import hashlib

import httpx
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, validator

from backend.models.analysis import (
    AIModelType, ProcessingStatus, AIModelConfig, AIModelResult,
    PlantIdentification, DiseaseIdentification, ConfidenceAggregation,
    ModelAggregationWeights, ProcessingError
)
from backend.models.disease import Disease
from backend.services.config import Settings
from backend.models.plant_classifier import PlantClassifier, PlantClassifierInference, create_plant_classifier
from backend.models.disease_detector import DiseaseDetector, DiseaseDetectorInference, create_disease_detector
from backend.models.ensemble_predictor import EnsemblePredictor, EnsembleMethod, create_ensemble_predictor

logger = logging.getLogger(__name__)


class AIResponse(BaseModel):
    """Standardized response from AI models"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: int
    model_type: AIModelType


class GrokRequest(BaseModel):
    """Request format for Grok AI API"""
    image_base64: str
    prompt: str = Field(..., description="Analysis prompt for the model")
    max_tokens: int = Field(1000, description="Maximum response tokens")
    temperature: float = Field(0.1, description="Response randomness")

    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('Temperature must be between 0 and 2')
        return v


class GrokResponse(BaseModel):
    """Response format from Grok AI API"""
    plant_name: Optional[str] = None
    plant_confidence: Optional[float] = None
    disease_name: Optional[str] = None
    disease_confidence: Optional[float] = None
    severity: Optional[str] = None
    symptoms_detected: List[str] = []
    alternative_plants: List[Dict[str, float]] = []
    alternative_diseases: List[Dict[str, float]] = []
    scientific_name: Optional[str] = None

    @validator('plant_confidence', 'disease_confidence')
    def validate_confidence(cls, v):
        if v is not None and not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v


class PlantNetRequest(BaseModel):
    """Request format for PlantNet API"""
    image_data: bytes
    organs: List[str] = ["leaf"]  # Focus on leaf identification
    include_related_images: bool = False
    nb_results: int = 10


class PlantNetResponse(BaseModel):
    """Response format from PlantNet API"""
    success: bool
    results: List[Dict[str, Any]] = []
    scientific_name: Optional[str] = None
    common_names: List[str] = []
    confidence: float = 0.0
    family: Optional[str] = None
    genus: Optional[str] = None


class AIService:
    """AI Service Integration Manager"""

    def __init__(self, settings: Settings):
        """Initialize AI service with settings"""
        self.settings = settings
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        self.model_configs = {
            AIModelType.GROK: AIModelConfig(
                model_type=AIModelType.GROK,
                endpoint_url=settings.GROK_ENDPOINT,
                api_key_required=True,
                max_retries=settings.GROK_MAX_RETRIES,
                timeout_seconds=settings.GROK_TIMEOUT_SECONDS,
                confidence_threshold=settings.MIN_PLANT_CONFIDENCE,
                enabled=bool(settings.GROK_API_KEY)
            ),
            AIModelType.PLANTNET: AIModelConfig(
                model_type=AIModelType.PLANTNET,
                endpoint_url=settings.PLANTNET_ENDPOINT,
                api_key_required=True,
                max_retries=settings.PLANTNET_MAX_RETRIES,
                timeout_seconds=settings.PLANTNET_TIMEOUT_SECONDS,
                confidence_threshold=settings.MIN_PLANT_CONFIDENCE,
                enabled=bool(settings.PLANTNET_API_KEY)
            ),
            AIModelType.INATURALIST: AIModelConfig(
                model_type=AIModelType.INATURALIST,
                endpoint_url=settings.INATURALIST_ENDPOINT,
                api_key_required=False,
                max_retries=settings.INATURALIST_MAX_RETRIES,
                timeout_seconds=settings.INATURALIST_TIMEOUT_SECONDS,
                confidence_threshold=settings.MIN_PLANT_CONFIDENCE,
                enabled=True
            )
        }
        self.aggregation_weights = settings.get_aggregation_weights()

        logger.info(f"AIService initialized with {len([c for c in self.model_configs.values() if c.enabled])} enabled models")

    async def initialize(self):
        """Initialize AI services and perform health checks"""
        logger.info("Initializing AI services...")

        health_status = {}
        for model_type, config in self.model_configs.items():
            if config.enabled:
                try:
                    if model_type == AIModelType.GROK:
                        health_status[model_type.value] = await self.check_grok_health()
                    elif model_type == AIModelType.PLANTNET:
                        health_status[model_type.value] = await self.check_plantnet_health()
                    elif model_type == AIModelType.INATURALIST:
                        health_status[model_type.value] = await self.check_inaturalist_health()

                    logger.info(f"{model_type.value} health check: {health_status[model_type.value]}")
                except Exception as e:
                    health_status[model_type.value] = False
                    logger.warning(f"{model_type.value} health check failed: {str(e)}")
            else:
                health_status[model_type.value] = False
                logger.info(f"{model_type.value} is disabled")

        logger.info(f"AI services initialization complete. Health status: {health_status}")
        return health_status

    async def cleanup(self):
        """Clean up resources"""
        await self.client.aclose()
        logger.info("AI service cleanup complete")

    def image_to_base64(self, image: Union[bytes, Image.Image]) -> str:
        """Convert image to base64 string"""
        try:
            if isinstance(image, Image.Image):
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=85)
                image_bytes = buffer.getvalue()
            else:
                image_bytes = image

            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to convert image to base64: {str(e)}")
            raise

    async def call_grok_api(self, image: Union[bytes, Image.Image], prompt_override: Optional[str] = None) -> GrokResponse:
        """Call Grok AI API for plant and disease analysis"""
        config = self.model_configs[AIModelType.GROK]
        if not config.enabled:
            raise ValueError("Grok API is not enabled or missing API key")

        # Default prompt for plant disease analysis
        default_prompt = """
        Analyze this plant leaf image and provide the following information in JSON format:

        {
            "plant_name": "Common name of the plant",
            "plant_confidence": 0.95,
            "scientific_name": "Scientific name of the plant",
            "disease_name": "Name of any disease detected (or null if healthy)",
            "disease_confidence": 0.87,
            "severity": "mild/moderate/severe (or null if healthy)",
            "symptoms_detected": ["list of visible symptoms"],
            "alternative_plants": [{"name": "Alternative plant 1", "confidence": 0.3}],
            "alternative_diseases": [{"name": "Alternative disease 1", "confidence": 0.2}]
        }

        Focus on accuracy for both plant identification and disease detection.
        Provide realistic confidence scores between 0.0 and 1.0.
        If no disease is detected, set disease_name to null and explain symptoms are normal.
        """

        prompt = prompt_override or default_prompt

        request_data = GrokRequest(
            image_base64=self.image_to_base64(image),
            prompt=prompt,
            max_tokens=1000,
            temperature=0.1
        )

        headers = {
            "Authorization": f"Bearer {self.settings.GROK_API_KEY}",
            "Content-Type": "application/json"
        }

        for attempt in range(config.max_retries + 1):
            try:
                start_time = time.time()

                response = await self.client.post(
                    config.endpoint_url,
                    json=request_data.dict(),
                    headers=headers,
                    timeout=config.timeout_seconds
                )

                processing_time_ms = int((time.time() - start_time) * 1000)

                if response.status_code == 200:
                    try:
                        # Parse JSON response
                        response_data = response.json()

                        # Extract the content from Grok's response format
                        content = response_data.get('content', response_data.get('choices', [{}])[0].get('message', {}).get('content', '{}'))

                        # Try to parse as JSON
                        try:
                            parsed_content = json.loads(content)
                        except:
                            # If JSON parsing fails, try to extract from text
                            import re
                            json_match = re.search(r'\{.*\}', content, re.DOTALL)
                            if json_match:
                                parsed_content = json.loads(json_match.group())
                            else:
                                raise ValueError("Could not parse response as JSON")

                        return GrokResponse(
                            plant_name=parsed_content.get('plant_name'),
                            plant_confidence=parsed_content.get('plant_confidence'),
                            disease_name=parsed_content.get('disease_name'),
                            disease_confidence=parsed_content.get('disease_confidence'),
                            severity=parsed_content.get('severity'),
                            symptoms_detected=parsed_content.get('symptoms_detected', []),
                            alternative_plants=parsed_content.get('alternative_plants', []),
                            alternative_diseases=parsed_content.get('alternative_diseases', []),
                            scientific_name=parsed_content.get('scientific_name')
                        )

                    except Exception as parse_error:
                        logger.error(f"Failed to parse Grok response: {str(parse_error)}")
                        logger.debug(f"Raw response: {response.text}")
                        raise ValueError(f"Failed to parse Grok response: {str(parse_error)}")

                else:
                    error_msg = f"Grok API error: {response.status_code} - {response.text}"
                    logger.warning(error_msg)

                    if attempt < config.max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        await asyncio.sleep(wait_time)
                        continue

                    raise ValueError(error_msg)

            except httpx.TimeoutException:
                error_msg = "Grok API timeout"
                if attempt < config.max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise ValueError(error_msg)

            except Exception as e:
                error_msg = f"Grok API call failed: {str(e)}"
                if attempt < config.max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise ValueError(error_msg)

        raise ValueError(f"Grok API failed after {config.max_retries + 1} attempts")

    async def call_plantnet_api(self, image: Union[bytes, Image.Image]) -> PlantNetResponse:
        """Call PlantNet API for specialized plant identification"""
        config = self.model_configs[AIModelType.PLANTNET]
        if not config.enabled:
            raise ValueError("PlantNet API is not enabled or missing API key")

        headers = {
            "Authorization": f"Bearer {self.settings.PLANTNET_API_KEY}",
        }

        files = {
            'images': ('image.jpg', image if isinstance(image, bytes) else io.BytesIO(), 'image/jpeg')
        }

        data = {
            'organs': 'leaf',
            'include-related-images': 'false',
            'nb-results': '10'
        }

        for attempt in range(config.max_retries + 1):
            try:
                start_time = time.time()

                response = await self.client.post(
                    config.endpoint_url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=config.timeout_seconds
                )

                processing_time_ms = int((time.time() - start_time) * 1000)

                if response.status_code == 200:
                    try:
                        response_data = response.json()

                        # Extract top result
                        results = response_data.get('results', [])
                        if results:
                            top_result = results[0]
                            scientific_name = top_result.get('species', {}).get('scientificNameWithoutAuthor')
                            common_names = [name.get('commonName') for name in top_result.get('species', {}).get('commonNames', []) if name.get('commonName')]
                            score = top_result.get('score', 0.0)
                            family = top_result.get('species', {}).get('family', {}).get('scientificNameWithoutAuthor')
                            genus = top_result.get('species', {}).get('genus', {}).get('scientificNameWithoutAuthor')

                            return PlantNetResponse(
                                success=True,
                                results=results,
                                scientific_name=scientific_name,
                                common_names=common_names,
                                confidence=score,
                                family=family,
                                genus=genus
                            )
                        else:
                            return PlantNetResponse(success=True, results=[], confidence=0.0)

                    except Exception as parse_error:
                        logger.error(f"Failed to parse PlantNet response: {str(parse_error)}")
                        raise ValueError(f"Failed to parse PlantNet response: {str(parse_error)}")

                else:
                    error_msg = f"PlantNet API error: {response.status_code} - {response.text}"
                    logger.warning(error_msg)

                    if attempt < config.max_retries:
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue

                    raise ValueError(error_msg)

            except httpx.TimeoutException:
                error_msg = "PlantNet API timeout"
                if attempt < config.max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise ValueError(error_msg)

            except Exception as e:
                error_msg = f"PlantNet API call failed: {str(e)}"
                if attempt < config.max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise ValueError(error_msg)

        raise ValueError(f"PlantNet API failed after {config.max_retries + 1} attempts")

    async def call_inaturalist_api(self, image: Union[bytes, Image.Image]) -> Dict[str, Any]:
        """Call iNaturalist API for species identification"""
        config = self.model_configs[AIModelType.INATURALIST]
        if not config.enabled:
            raise ValueError("iNaturalist API is not enabled")

        # Note: iNaturalist's vision API may have different endpoints
        # This is a placeholder implementation
        headers = {
            "Content-Type": "multipart/form-data"
        }

        files = {
            'image': ('image.jpg', image if isinstance(image, bytes) else io.BytesIO(), 'image/jpeg')
        }

        try:
            start_time = time.time()

            # This would be the actual iNaturalist endpoint
            response = await self.client.post(
                config.endpoint_url,
                headers=headers,
                files=files,
                timeout=config.timeout_seconds
            )

            processing_time_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "processing_time_ms": processing_time_ms
                }
            else:
                logger.warning(f"iNaturalist API error: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "processing_time_ms": processing_time_ms
                }

        except Exception as e:
            logger.error(f"iNaturalist API call failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": 0
            }

    async def analyze_with_local_models(self, image: Union[bytes, Image.Image]) -> List[AIModelResult]:
        """Analyze image with local models only"""
        results = []

        # Plant classification with local model
        if self.plant_classifier:
            try:
                start_time = time.time()

                # Create inference wrapper
                if not hasattr(self.plant_classifier, 'predict'):
                    from backend.models.plant_classifier import PlantClassifierInference
                    plant_classifier_inference = PlantClassifierInference(
                        model=self.plant_classifier,
                        device='cpu',
                        confidence_threshold=self.settings.MIN_PLANT_CONFIDENCE
                    )
                else:
                    plant_classifier_inference = self.plant_classifier

                # Get prediction
                plant_result = plant_classifier_inference.predict_image(image, top_k=5)

                processing_time_ms = int((time.time() - start_time) * 1000)

                if plant_result['predictions']:
                    top_prediction = plant_result['predictions'][0]

                    plant_identification = PlantIdentification(
                        plant_name=top_prediction['species'],
                        confidence=top_prediction['confidence'],
                        scientific_name=top_prediction.get('scientific_name'),
                        alternative_identifications=[
                            {
                                "plant_name": alt.get('species'),
                                "confidence": alt.get('confidence'),
                                "scientific_name": alt.get('scientific_name')
                            }
                            for alt in plant_result['top_k_predictions'][1:4]
                        ],
                        model_used=AIModelType.PLANT_CLASSIFIER,
                        processing_time_ms=processing_time_ms
                    )

                    results.append(AIModelResult(
                        model_type=AIModelType.PLANT_CLASSIFIER,
                        plant_identification=plant_identification,
                        disease_identification=None,
                        raw_response=plant_result,
                        success=True,
                        processing_time_ms=processing_time_ms,
                        timestamp=time.time()
                    ))
                else:
                    # No predictions returned
                    results.append(AIModelResult(
                        model_type=AIModelType.PLANT_CLASSIFIER,
                        plant_identification=None,
                        disease_identification=None,
                        raw_response=plant_result,
                        success=False,
                        error_message="No plant predictions returned",
                        processing_time_ms=processing_time_ms,
                        timestamp=time.time()
                    ))

                logger.info(f"Local plant classification completed in {processing_time_ms}ms")

            except Exception as e:
                processing_time_ms = int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
                logger.error(f"Local plant classification failed: {str(e)}")
                results.append(AIModelResult(
                    model_type=AIModelType.PLANT_CLASSIFIER,
                    plant_identification=None,
                    disease_identification=None,
                    raw_response=None,
                    success=False,
                    error_message=str(e),
                    processing_time_ms=processing_time_ms,
                    timestamp=time.time()
                ))

        # Disease detection with local model
        if self.disease_detector:
            try:
                start_time = time.time()

                # Create inference wrapper
                if not hasattr(self.disease_detector, 'predict'):
                    from backend.models.disease_detector import DiseaseDetectorInference
                    disease_detector_inference = DiseaseDetectorInference(
                        model=self.disease_detector,
                        device='cpu',
                        confidence_threshold=self.settings.MIN_DISEASE_CONFIDENCE
                    )
                else:
                    disease_detector_inference = self.disease_detector

                # Get prediction with Grad-CAM
                disease_result = disease_detector_inference.predict_image(image, generate_grad_cam=True)

                processing_time_ms = int((time.time() - start_time) * 1000)

                if disease_result['detections']:
                    top_detection = disease_result['detections'][0]

                    disease_identification = DiseaseIdentification(
                        disease_name=top_detection['disease'],
                        confidence=top_detection['confidence'],
                        disease_id=None,  # Will be resolved by ensemble
                        symptoms_detected=[],  # Will be extracted from image processing
                        severity=top_detection.get('severity'),
                        scientific_name=f"{top_detection['disease']} pathogen" if top_detection['disease'] != 'healthy' else "Healthy plant",
                        model_used=AIModelType.DISEASE_DETECTOR,
                        processing_time_ms=processing_time_ms
                    )

                    results.append(AIModelResult(
                        model_type=AIModelType.DISEASE_DETECTOR,
                        plant_identification=None,
                        disease_identification=disease_identification,
                        raw_response=disease_result,
                        success=True,
                        processing_time_ms=processing_time_ms,
                        timestamp=time.time()
                    ))
                else:
                    # No detections returned
                    results.append(AIModelResult(
                        model_type=AIModelType.DISEASE_DETECTOR,
                        plant_identification=None,
                        disease_identification=None,
                        raw_response=disease_result,
                        success=False,
                        error_message="No disease detections returned",
                        processing_time_ms=processing_time_ms,
                        timestamp=time.time()
                    ))

                logger.info(f"Local disease detection completed in {processing_time_ms}ms")

            except Exception as e:
                processing_time_ms = int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
                logger.error(f"Local disease detection failed: {str(e)}")
                results.append(AIModelResult(
                    model_type=AIModelType.DISEASE_DETECTOR,
                    plant_identification=None,
                    disease_identification=None,
                    raw_response=None,
                    success=False,
                    error_message=str(e),
                    processing_time_ms=processing_time_ms,
                    timestamp=time.time()
                ))

        # Ensemble prediction with all models
        if self.ensemble_predictor and (self.plant_classifier or self.disease_detector):
            try:
                start_time = time.time()

                # Use ensemble predictor for combined results
                ensemble_prediction = await self.ensemble_predictor.predict_with_all_models(image, self)

                processing_time_ms = int((time.time() - start_time) * 1000)

                # Convert ensemble prediction to AI model result
                ensemble_plant_identification = None
                ensemble_disease_identification = None

                if ensemble_prediction.plant_name:
                    ensemble_plant_identification = PlantIdentification(
                        plant_name=ensemble_prediction.plant_name,
                        confidence=ensemble_prediction.plant_confidence,
                        scientific_name=None,  # Could be mapped from plant database
                        alternative_identifications=[
                            {
                                "plant_name": alt[0],
                                "confidence": alt[1],
                                "scientific_name": None
                            }
                            for alt in ensemble_prediction.plant_alternatives
                        ],
                        model_used=AIModelType.ENSEMBLE,
                        processing_time_ms=processing_time_ms
                    )

                if ensemble_prediction.disease_name:
                    ensemble_disease_identification = DiseaseIdentification(
                        disease_name=ensemble_prediction.disease_name,
                        confidence=ensemble_prediction.disease_confidence,
                        disease_id=None,  # Will be resolved by disease service
                        symptoms_detected=[],  # Will be extracted from image processing
                        severity=ensemble_prediction.severity_level,
                        scientific_name=f"{ensemble_prediction.disease_name}" if ensemble_prediction.disease_name != 'healthy' else "Healthy plant",
                        model_used=AIModelType.ENSEMBLE,
                        processing_time_ms=processing_time_ms
                    )

                # Add ensemble result
                results.append(AIModelResult(
                    model_type=AIModelType.ENSEMBLE,
                    plant_identification=ensemble_plant_identification,
                    disease_identification=ensemble_disease_identification,
                    raw_response=ensemble_prediction.__dict__,
                    success=True,
                    processing_time_ms=processing_time_ms,
                    timestamp=time.time()
                ))

                logger.info(f"Local ensemble prediction completed in {processing_time_ms}ms")

            except Exception as e:
                processing_time_ms = int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
                logger.error(f"Local ensemble prediction failed: {str(e)}")
                results.append(AIModelResult(
                    model_type=AIModelType.ENSEMBLE,
                    plant_identification=None,
                    disease_identification=None,
                    raw_response=None,
                    success=False,
                    error_message=str(e),
                    processing_time_ms=processing_time_ms,
                    timestamp=time.time()
                ))

        return results

    async def analyze_with_all_models(self, image: Union[bytes, Image.Image]) -> List[AIModelResult]:
        """Analyze image with all available AI models including local models"""
        local_results = await self.analyze_with_local_models(image)
        external_results = await self.analyze_with_external_models(image)

        # Combine all results
        all_results = local_results + external_results

        logger.info(f"Total analysis completed: {len(local_results)} local, {len(external_results)} external models")

        return all_results

    async def analyze_with_external_models(self, image: Union[bytes, Image.Image]) -> List[AIModelResult]:
        """Analyze image with external API models only"""
        tasks = []

        # Create concurrent tasks for enabled external models
        if self.model_configs[AIModelType.GROK].enabled:
            tasks.append(self.analyze_with_grok(image))

        if self.model_configs[AIModelType.PLANTNET].enabled:
            tasks.append(self.analyze_with_plantnet(image))

        if self.model_configs[AIModelType.INATURALIST].enabled:
            tasks.append(self.analyze_with_inaturalist(image))

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return successful results
        model_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"External model analysis failed: {str(result)}")
            else:
                model_results.append(result)

        return model_results

    async def analyze_with_grad_cam(self, image: Union[bytes, Image.Image]) -> Dict[str, Any]:
        """Generate Grad-CAM visualization for disease detection"""
        try:
            # Initialize disease detector if not available
            if not hasattr(self, 'disease_detector') or self.disease_detector is None:
                return {"success": False, "error": "Disease detector model not available"}

            # Convert PIL Image to bytes if needed
            if isinstance(image, bytes):
                pil_image = Image.open(io.BytesIO(image))
            else:
                pil_image = image

            # Initialize disease detector inference
            from backend.models.disease_detector import DiseaseDetectorInference
            disease_detector_inference = DiseaseDetectorInference(
                model_path=getattr(self.settings, 'LOCAL_DISEASE_MODEL_PATH', 'models/disease_detector.pth'),
                device=getattr(self.settings, 'TORCH_DEVICE', 'cpu')
            )

            # Generate prediction with Grad-CAM
            result = disease_detector_inference.predict_image(
                pil_image,
                top_k_diseases=1,
                confidence_threshold=0.0,  # Get prediction even if low confidence
                generate_grad_cam=True
            )

            if result.success and result.grad_cam_heatmap is not None:
                # Convert Grad-CAM heatmap to base64 for JSON serialization
                heatmap_pil = result.grad_cam_heatmap

                # Resize heatmap to match original image dimensions
                heatmap_resized = heatmap_pil.resize(pil_image.size, Image.LANCZOS)

                # Create overlay visualization
                import cv2
                import numpy as np

                # Convert images to numpy arrays
                original_np = np.array(pil_image.convert('RGB'))
                heatmap_np = np.array(heatmap_resized)

                # Apply colormap to heatmap
                heatmap_colored = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

                # Blend original image with heatmap
                alpha = 0.6  # Transparency factor
                overlay = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0)

                # Convert overlay back to PIL Image
                overlay_pil = Image.fromarray(overlay)

                # Convert both images to base64
                def image_to_base64(img):
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    return base64.b64encode(buffer.getvalue()).decode('utf-8')

                # Encode original image
                original_base64 = image_to_base64(pil_image)

                # Encode heatmap
                heatmap_base64 = image_to_base64(heatmap_resized)

                # Encode overlay
                overlay_base64 = image_to_base64(overlay_pil)

                # Get prediction details
                prediction_details = {}
                if result.disease_predictions:
                    pred = result.disease_predictions[0]
                    prediction_details = {
                        "disease_name": pred.class_name,
                        "confidence": pred.confidence,
                        "severity": pred.severity_prediction,
                        "is_healthy": pred.class_name.lower() == 'healthy'
                    }

                return {
                    "success": True,
                    "original_image": original_base64,
                    "heatmap_image": heatmap_base64,
                    "overlay_image": overlay_base64,
                    "prediction": prediction_details,
                    "metadata": {
                        "model_version": result.model_version,
                        "device": result.device_used,
                        "processing_time": result.processing_time,
                        "heatmap_generated": True,
                        "image_size": pil_image.size
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to generate Grad-CAM visualization",
                    "details": result.error_message if hasattr(result, 'error_message') else "Unknown error"
                }

        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {str(e)}")
            return {
                "success": False,
                "error": f"Grad-CAM generation error: {str(e)}"
            }

    async def analyze_with_grok(self, image: Union[bytes, Image.Image]) -> AIModelResult:
        """Analyze image specifically with Grok AI"""
        start_time = time.time()

        try:
            grok_response = await self.call_grok_api(image)
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Convert to standardized format
            plant_identification = PlantIdentification(
                plant_name=grok_response.plant_name or "Unknown",
                confidence=grok_response.plant_confidence or 0.0,
                scientific_name=grok_response.scientific_name,
                alternative_identifications=grok_response.alternative_plants or [],
                model_used=AIModelType.GROK,
                processing_time_ms=processing_time_ms
            )

            disease_identification = None
            if grok_response.disease_name and grok_response.disease_confidence:
                disease_identification = DiseaseIdentification(
                    disease_name=grok_response.disease_name,
                    disease_id=None,  # Will be resolved later
                    confidence=grok_response.disease_confidence,
                    severity=grok_response.severity or "unknown",
                    symptoms_detected=grok_response.symptoms_detected or [],
                    model_used=AIModelType.GROK,
                    processing_time_ms=processing_time_ms
                )

            return AIModelResult(
                model_type=AIModelType.GROK,
                plant_identification=plant_identification,
                disease_identification=disease_identification,
                raw_response=grok_response.dict(),
                success=True,
                processing_time_ms=processing_time_ms,
                timestamp=time.time()
            )

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)

            return AIModelResult(
                model_type=AIModelType.GROK,
                plant_identification=None,
                disease_identification=None,
                raw_response=None,
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time_ms,
                timestamp=time.time()
            )

    async def analyze_with_plantnet(self, image: Union[bytes, Image.Image]) -> AIModelResult:
        """Analyze image specifically with PlantNet"""
        start_time = time.time()

        try:
            plantnet_response = await self.call_plantnet_api(image)
            processing_time_ms = int((time.time() - start_time) * 1000)

            if plantnet_response.success and plantnet_response.scientific_name:
                # Map to common name (would need a lookup table in practice)
                common_name = plantnet_response.common_names[0] if plantnet_response.common_names else plantnet_response.scientific_name

                plant_identification = PlantIdentification(
                    plant_name=common_name,
                    confidence=plantnet_response.confidence,
                    scientific_name=plantnet_response.scientific_name,
                    alternative_identifications=[],  # Could extract from results
                    model_used=AIModelType.PLANTNET,
                    processing_time_ms=processing_time_ms
                )
            else:
                plant_identification = PlantIdentification(
                    plant_name="Unknown",
                    confidence=0.0,
                    scientific_name=None,
                    alternative_identifications=[],
                    model_used=AIModelType.PLANTNET,
                    processing_time_ms=processing_time_ms
                )

            # PlantNet only does plant identification, not disease detection
            return AIModelResult(
                model_type=AIModelType.PLANTNET,
                plant_identification=plant_identification,
                disease_identification=None,
                raw_response=plantnet_response.dict(),
                success=True,
                processing_time_ms=processing_time_ms,
                timestamp=time.time()
            )

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)

            return AIModelResult(
                model_type=AIModelType.PLANTNET,
                plant_identification=None,
                disease_identification=None,
                raw_response=None,
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time_ms,
                timestamp=time.time()
            )

    async def analyze_with_inaturalist(self, image: Union[bytes, Image.Image]) -> AIModelResult:
        """Analyze image specifically with iNaturalist"""
        start_time = time.time()

        try:
            inaturalist_response = await self.call_inaturalist_api(image)
            processing_time_ms = int((time.time() - start_time) * 1000)

            # This would need to be adapted based on actual iNaturalist API response format
            if inaturalist_response.get("success"):
                data = inaturalist_response.get("data", {})

                plant_identification = PlantIdentification(
                    plant_name=data.get("common_name", "Unknown"),
                    confidence=data.get("confidence", 0.0),
                    scientific_name=data.get("scientific_name"),
                    alternative_identifications=data.get("alternatives", []),
                    model_used=AIModelType.INATURALIST,
                    processing_time_ms=processing_time_ms
                )
            else:
                plant_identification = PlantIdentification(
                    plant_name="Unknown",
                    confidence=0.0,
                    scientific_name=None,
                    alternative_identifications=[],
                    model_used=AIModelType.INATURALIST,
                    processing_time_ms=processing_time_ms
                )

            return AIModelResult(
                model_type=AIModelType.INATURALIST,
                plant_identification=plant_identification,
                disease_identification=None,
                raw_response=inaturalist_response,
                success=inaturalist_response.get("success", False),
                error_message=inaturalist_response.get("error"),
                processing_time_ms=processing_time_ms,
                timestamp=time.time()
            )

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)

            return AIModelResult(
                model_type=AIModelType.INATURALIST,
                plant_identification=None,
                disease_identification=None,
                raw_response=None,
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time_ms,
                timestamp=time.time()
            )

    def aggregate_results(self, model_results: List[AIModelResult]) -> ConfidenceAggregation:
        """Aggregate results from multiple AI models with confidence weighting"""
        if not model_results:
            raise ValueError("No model results to aggregate")

        # Collect plant identifications
        plant_confidences = {}
        plant_alternatives = {}

        for result in model_results:
            if result.plant_identification and result.success:
                plant_name = result.plant_identification.plant_name
                confidence = result.plant_identification.confidence

                # Apply weighting
                weight = self.aggregation_weights['plant_identification'].get(result.model_type.value, 0.5)
                weighted_confidence = confidence * weight

                if plant_name in plant_confidences:
                    plant_confidences[plant_name] += weighted_confidence
                    # Count occurrences for alternative consideration
                else:
                    plant_confidences[plant_name] = weighted_confidence

                # Collect alternatives
                for alt in result.plant_identification.alternative_identifications:
                    alt_name = alt.get('name', alt.get('plant_name', 'Unknown'))
                    alt_conf = alt.get('confidence', 0.0) * weight * 0.5  # Lower weight for alternatives

                    if alt_name not in plant_alternatives:
                        plant_alternatives[alt_name] = []
                    plant_alternatives[alt_name].append(alt_conf)

        # Find primary plant identification
        primary_plant_name = max(plant_confidences.keys(), key=lambda k: plant_confidences[k])
        plant_confidence = min(1.0, plant_confidences[primary_plant_name])

        # Prepare alternatives
        plant_alternatives_list = []
        for alt_name, confs in plant_alternatives.items():
            if alt_name != primary_plant_name:
                avg_conf = sum(confs) / len(confs)
                plant_alternatives_list.append({"name": alt_name, "confidence": avg_conf})

        # Sort alternatives by confidence
        plant_alternatives_list.sort(key=lambda x: x["confidence"], reverse=True)
        plant_alternatives_list = plant_alternatives_list[:3]  # Top 3 alternatives

        # Collect disease identifications
        disease_confidences = {}
        disease_alternatives = {}

        for result in model_results:
            if result.disease_identification and result.success:
                disease_name = result.disease_identification.disease_name
                confidence = result.disease_identification.confidence

                if disease_name and disease_name.lower() not in ["null", "none", "healthy"]:
                    weight = self.aggregation_weights['disease_identification'].get(result.model_type.value, 0.5)
                    weighted_confidence = confidence * weight

                    if disease_name in disease_confidences:
                        disease_confidences[disease_name] += weighted_confidence
                    else:
                        disease_confidences[disease_name] = weighted_confidence

        # Check for model consensus
        unique_plants = len(plant_confidences)
        unique_diseases = len(disease_confidences)
        models_agreed = unique_plants <= 2  # Consider consensus if models are close

        # Primary disease identification
        primary_disease_name = None
        disease_confidence = None

        if disease_confidences:
            primary_disease_name = max(disease_confidences.keys(), key=lambda k: disease_confidences[k])
            disease_confidence = min(1.0, disease_confidences[primary_disease_name])

        # Apply consensus bonus
        consensus_bonus = self.aggregation_weights.get('consensus_bonus', 0.0)
        if models_agreed and consensus_bonus > 0:
            plant_confidence = min(1.0, plant_confidence + consensus_bonus)
            if disease_confidence:
                disease_confidence = min(1.0, disease_confidence + consensus_bonus)

        # Prepare disease alternatives
        disease_alternatives_list = []
        for alt_name, confs in disease_alternatives.items():
            if alt_name != primary_disease_name:
                avg_conf = sum(confs) / len(confs)
                disease_alternatives_list.append({"name": alt_name, "confidence": avg_conf})

        disease_alternatives_list.sort(key=lambda x: x["confidence"], reverse=True)
        disease_alternatives_list = disease_alternatives_list[:3]  # Top 3 alternatives

        return ConfidenceAggregation(
            primary_plant_name=primary_plant_name,
            plant_confidence=plant_confidence,
            plant_alternatives=plant_alternatives_list,
            primary_disease_name=primary_disease_name,
            disease_confidence=disease_confidence,
            disease_alternatives=disease_alternatives_list,
            models_agreed=models_agreed,
            aggregation_method="weighted_average_with_consensus"
        )

    async def get_available_models(self) -> List[str]:
        """Get list of available AI models"""
        return [model_type.value for model_type, config in self.model_configs.items() if config.enabled]

    async def check_grok_health(self) -> bool:
        """Check Grok API health"""
        try:
            if not self.model_configs[AIModelType.GROK].enabled:
                return False

            # Simple health check - could be a dedicated health endpoint
            headers = {"Authorization": f"Bearer {self.settings.GROK_API_KEY}"}
            response = await self.client.get(
                f"{self.settings.GROK_ENDPOINT}/health",
                headers=headers,
                timeout=5.0
            )
            return response.status_code < 500
        except:
            return False

    async def check_plantnet_health(self) -> bool:
        """Check PlantNet API health"""
        try:
            if not self.model_configs[AIModelType.PLANTNET].enabled:
                return False

            headers = {"Authorization": f"Bearer {self.settings.PLANTNET_API_KEY}"}
            response = await self.client.get(
                f"{self.settings.PLANTNET_ENDPOINT}/health",
                headers=headers,
                timeout=5.0
            )
            return response.status_code < 500
        except:
            return False

    async def check_inaturalist_health(self) -> bool:
        """Check iNaturalist API health"""
        try:
            response = await self.client.get(
                f"{self.settings.INATURALIST_ENDPOINT}/health",
                timeout=5.0
            )
            return response.status_code < 500
        except:
            return False

    def get_model_statistics(self) -> Dict[str, Any]:
        """Get AI service statistics"""
        return {
            "enabled_models": [model_type.value for model_type, config in self.model_configs.items() if config.enabled],
            "total_models": len(self.model_configs),
            "aggregation_weights": self.aggregation_weights,
            "api_keys_configured": {
                "grok": bool(self.settings.GROK_API_KEY),
                "plantnet": bool(self.settings.PLANTNET_API_KEY),
                "inaturalist": True  # iNaturalist doesn't require API key
            }
        }