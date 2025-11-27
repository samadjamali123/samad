"""
Image Processing Service

This module handles all image processing tasks including validation, quality assessment,
preprocessing, and format standardization for AI model consumption.
"""

import os
import io
import logging
import tempfile
import time
from typing import Tuple, List, Optional, Dict, Any, Union
from pathlib import Path
import hashlib

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage import exposure, filters, measure, morphology
from pydantic import BaseModel, Field, validator

from backend.models.analysis import (
    ImageMetadata, QualityMetrics, ImageAnalysisResult,
    ImageFormat, ProcessingError
)
from backend.services.config import Settings

logger = logging.getLogger(__name__)


class ProcessingConfig(BaseModel):
    """Configuration for image processing"""
    max_file_size_mb: float = Field(..., description="Maximum file size in MB")
    min_quality_score: float = Field(..., description="Minimum acceptable quality score")
    min_dimension: int = Field(..., description="Minimum image dimension")
    max_dimension: int = Field(..., description="Maximum image dimension")
    supported_formats: List[str] = Field(..., description="Supported image formats")
    jpeg_quality: int = Field(..., description="JPEG compression quality")
    png_compression: int = Field(..., description="PNG compression level")
    enable_sharpening: bool = Field(..., description="Enable image sharpening")
    enable_noise_reduction: bool = Field(..., description="Enable noise reduction")


class ImageProcessor:
    """Advanced image processing service for plant disease detection"""

    def __init__(self, settings: Settings):
        """Initialize image processor with settings"""
        self.settings = settings
        self.config = ProcessingConfig(
            max_file_size_mb=settings.MAX_FILE_SIZE_MB,
            min_quality_score=settings.MIN_IMAGE_QUALITY_SCORE,
            min_dimension=settings.MIN_IMAGE_SIZE,
            max_dimension=settings.MAX_IMAGE_SIZE,
            supported_formats=settings.SUPPORTED_FORMATS_LIST,
            jpeg_quality=settings.JPEG_QUALITY,
            png_compression=settings.PNG_COMPRESSION,
            enable_sharpening=settings.ENABLE_IMAGE_SHARPENING,
            enable_noise_reduction=settings.ENABLE_NOISE_REDUCTION
        )
        self.temp_dir = Path(settings.TEMP_DIR_ABSOLUTE)
        self.temp_dir.mkdir(exist_ok=True)

        logger.info(f"ImageProcessor initialized with temp directory: {self.temp_dir}")

    def validate_file_format(self, file: Union[bytes, io.BytesIO]) -> bool:
        """Validate that the file is a supported image format"""
        try:
            # Try to open with PIL to verify format
            if isinstance(file, bytes):
                image = Image.open(io.BytesIO(file))
            else:
                file.seek(0)
                image = Image.open(file)
                file.seek(0)

            # Check format
            format_name = image.format.lower() if image.format else None
            return format_name in self.config.supported_formats

        except Exception as e:
            logger.warning(f"Format validation failed: {str(e)}")
            return False

    def validate_file_size(self, file: Union[bytes, io.BytesIO]) -> Tuple[bool, float]:
        """Validate file size against maximum limit"""
        try:
            if isinstance(file, bytes):
                size_bytes = len(file)
            else:
                file.seek(0, os.SEEK_END)
                size_bytes = file.tell()
                file.seek(0)

            size_mb = size_bytes / (1024 * 1024)
            return size_mb <= self.config.max_file_size_mb, size_mb

        except Exception as e:
            logger.warning(f"Size validation failed: {str(e)}")
            return False, 0.0

    def extract_metadata(self, file: Union[bytes, io.BytesIO]) -> Optional[ImageMetadata]:
        """Extract comprehensive metadata from image file"""
        try:
            if isinstance(file, bytes):
                image = Image.open(io.BytesIO(file))
                size_bytes = len(file)
                image_data = file
            else:
                file.seek(0)
                image = Image.open(file)
                file.seek(0)
                # Read bytes from file-like object
                image_data = file.read()
                file.seek(0)
                size_bytes = len(image_data)

            # Basic metadata
            width, height = image.size
            format_name = image.format.lower() if image.format else 'unknown'
            size_mb = size_bytes / (1024 * 1024)
            aspect_ratio = width / height if height > 0 else 1.0
            color_space = image.mode if hasattr(image, 'mode') else 'RGB'
            bits_per_channel = getattr(image, 'bits', 8)

            return ImageMetadata(
                filename=getattr(file, 'name', 'upload.jpg') if hasattr(file, 'name') else 'upload',
                format=ImageFormat(format_name) if format_name in self.config.supported_formats else ImageFormat.JPG,
                size_bytes=size_bytes,
                size_mb=round(size_mb, 2),
                dimensions=(width, height),
                aspect_ratio=round(aspect_ratio, 3),
                color_space=color_space,
                bits_per_channel=bits_per_channel
            )

        except Exception as e:
            logger.error(f"Metadata extraction failed: {str(e)}")
            return None

    def assess_image_quality(self, image_array: np.ndarray) -> QualityMetrics:
        """Comprehensive quality assessment using multiple metrics"""
        try:
            # Convert to grayscale for analysis
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array

            # Sharpness assessment using Laplacian variance
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, sharpness / 500.0)  # Normalize to 0-1

            # Brightness assessment
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0  # Optimal is 128

            # Contrast assessment using standard deviation
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 64.0)  # Normalize to 0-1

            # Noise assessment using median filter difference
            noise = np.mean(cv2.absdiff(gray, cv2.medianBlur(gray, 3)))
            noise_score = 1.0 - min(1.0, noise / 50.0)  # Lower is better, invert

            # Color balance assessment
            if len(image_array.shape) == 3:
                r_mean = np.mean(image_array[:, :, 0])
                g_mean = np.mean(image_array[:, :, 1])
                b_mean = np.mean(image_array[:, :, 2])
                color_balance_score = 1.0 - (max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)) / 255.0
            else:
                color_balance_score = 0.8  # Default for grayscale

            # Composition assessment - focus on central area
            height, width = gray.shape
            center_x, center_y = width // 2, height // 2
            center_region = gray[center_y-height//4:center_y+height//4, center_x-width//4:center_x+width//4]
            edge_region = np.concatenate([
                gray[:height//4, :width//4],  # Top-left corner
                gray[:height//4, -width//4:],  # Top-right corner
                gray[-height//4:, :width//4],  # Bottom-left corner
                gray[-height//4:, -width//4:]   # Bottom-right corner
            ])
            composition_score = min(1.0, (np.mean(center_region) - np.mean(edge_region)) / 64.0 + 0.5)

            # Leaf coverage estimation using segmentation
            try:
                # Simple thresholding to estimate leaf area
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                leaf_pixels = np.sum(binary > 0)
                total_pixels = binary.size
                leaf_coverage = leaf_pixels / total_pixels
                leaf_coverage_score = min(1.0, leaf_coverage * 3)  # Scale up for better scoring
            except:
                leaf_coverage_score = 0.7  # Default if segmentation fails

            # Calculate overall score
            scores = [sharpness_score, brightness_score, contrast_score, noise_score,
                     color_balance_score, composition_score, leaf_coverage_score]
            overall_score = np.mean(scores)

            return QualityMetrics(
                overall_score=round(float(overall_score), 3),
                sharpness_score=round(float(sharpness_score), 3),
                brightness_score=round(float(brightness_score), 3),
                contrast_score=round(float(contrast_score), 3),
                noise_level=round(float(1.0 - noise_score), 3),
                color_balance=round(float(color_balance_score), 3),
                composition_score=round(float(composition_score), 3),
                leaf_coverage=round(float(leaf_coverage_score), 3)
            )

        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            # Return default metrics on failure
            return QualityMetrics(
                overall_score=0.5,
                sharpness_score=0.5,
                brightness_score=0.5,
                contrast_score=0.5,
                noise_level=0.5,
                color_balance=0.5,
                composition_score=0.5,
                leaf_coverage=0.5
            )

    def preprocess_image(self, image: Image.Image, quality_metrics: QualityMetrics) -> Tuple[Image.Image, List[str]]:
        """Apply preprocessing based on quality assessment"""
        applied_steps = []
        processed_image = image.copy()

        try:
            # Resize to optimal dimensions
            current_size = processed_image.size
            max_dim = max(current_size)
            min_dim = min(current_size)

            if max_dim > self.config.max_dimension:
                # Scale down to max dimension
                scale_factor = self.config.max_dimension / max_dim
                new_size = (int(current_size[0] * scale_factor), int(current_size[1] * scale_factor))
                processed_image = processed_image.resize(new_size, Image.Resampling.LANCZOS)
                applied_steps.append(f"Resized from {current_size} to {new_size}")

            elif min_dim < self.config.min_dimension:
                # Scale up to minimum dimension
                scale_factor = self.config.min_dimension / min_dim
                new_size = (int(current_size[0] * scale_factor), int(current_size[1] * scale_factor))
                processed_image = processed_image.resize(new_size, Image.Resampling.LANCZOS)
                applied_steps.append(f"Resized from {current_size} to {new_size}")

            # Convert to RGB if not already
            if processed_image.mode != 'RGB':
                processed_image = processed_image.convert('RGB')
                applied_steps.append(f"Converted from {processed_image.mode} to RGB")

            # Apply noise reduction if enabled and needed
            if self.config.enable_noise_reduction and quality_metrics.noise_level > 0.7:
                processed_image = processed_image.filter(ImageFilter.MedianFilter(size=3))
                applied_steps.append("Applied median noise reduction")

            # Enhance contrast if needed
            if quality_metrics.contrast_score < 0.6:
                enhancer = ImageEnhance.Contrast(processed_image)
                processed_image = enhancer.enhance(1.2)
                applied_steps.append("Enhanced contrast")

            # Enhance brightness if needed
            if quality_metrics.brightness_score < 0.6:
                enhancer = ImageEnhance.Brightness(processed_image)
                processed_image = enhancer.enhance(1.1)
                applied_steps.append("Enhanced brightness")

            # Apply sharpening if enabled and needed
            if self.config.enable_sharpening and quality_metrics.sharpness_score < 0.7:
                processed_image = processed_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                applied_steps.append("Applied unsharp mask")

            # Normalize color balance
            if quality_metrics.color_balance < 0.7:
                # Simple color balance using histogram equalization
                if processed_image.mode == 'RGB':
                    processed_array = np.array(processed_image)
                    for i in range(3):  # RGB channels
                        processed_array[:, :, i] = exposure.equalize_hist(processed_array[:, :, i]) * 255
                    processed_image = Image.fromarray(processed_array.astype(np.uint8))
                    applied_steps.append("Normalized color balance")

            return processed_image, applied_steps

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return image, []  # Return original image if preprocessing fails

    def generate_recommendations(self, quality_metrics: QualityMetrics, metadata: ImageMetadata) -> List[str]:
        """Generate improvement recommendations based on quality assessment"""
        recommendations = []

        if quality_metrics.overall_score < self.config.min_quality_score:
            if quality_metrics.sharpness_score < 0.6:
                recommendations.append("Image appears blurry. Use higher focus or increase camera stability.")

            if quality_metrics.brightness_score < 0.6:
                recommendations.append("Image is too dark or too bright. Improve lighting conditions.")

            if quality_metrics.contrast_score < 0.6:
                recommendations.append("Image lacks contrast. Ensure even lighting and avoid overexposure.")

            if quality_metrics.noise_level > 0.7:
                recommendations.append("Image has excessive noise. Use lower ISO settings or better lighting.")

            if quality_metrics.color_balance < 0.6:
                recommendations.append("Color balance seems off. Check white balance settings.")

            if quality_metrics.composition_score < 0.6:
                recommendations.append("Image composition could be improved. Center the leaf and ensure adequate framing.")

            if quality_metrics.leaf_coverage < 0.5:
                recommendations.append("Leaf doesn't occupy enough of the frame. Get closer to the leaf.")

            if metadata.dimensions[0] < self.config.min_dimension or metadata.dimensions[1] < self.config.min_dimension:
                recommendations.append(f"Image resolution is too low. Minimum {self.config.min_dimension}x{self.config.min_dimension} pixels required.")

        if metadata.size_mb > self.config.max_file_size_mb * 0.8:
            recommendations.append("File size is large. Consider using JPEG format with moderate compression.")

        return recommendations

    def save_temp_image(self, image: Image.Image, prefix: str = "processed_") -> str:
        """Save image to temporary file and return file path"""
        try:
            # Generate unique filename
            timestamp = int(time.time() * 1000)
            file_hash = hashlib.md5(image.tobytes()).hexdigest()[:8]
            filename = f"{prefix}{timestamp}_{file_hash}.jpg"
            filepath = self.temp_dir / filename

            # Save with optimized settings
            image.save(
                filepath,
                format='JPEG',
                quality=self.config.jpeg_quality,
                optimize=True
            )

            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save temporary image: {str(e)}")
            raise

    async def analyze_image(self, file: Union[bytes, io.BytesIO]) -> ImageAnalysisResult:
        """Complete image analysis including validation, quality assessment, and preprocessing"""
        start_time = time.time()

        try:
            # Extract metadata
            metadata = self.extract_metadata(file)
            if not metadata:
                raise ValueError("Failed to extract image metadata")

            # Validate file format
            if not self.validate_file_format(file):
                raise ValueError(f"Unsupported image format. Supported formats: {', '.join(self.config.supported_formats)}")

            # Validate file size
            size_valid, actual_size_mb = self.validate_file_size(file)
            if not size_valid:
                raise ValueError(f"File too large. Maximum size: {self.config.max_file_size_mb}MB, Actual: {actual_size_mb:.2f}MB")

            # Convert to PIL Image for processing
            if isinstance(file, bytes):
                image = Image.open(io.BytesIO(file))
            else:
                file.seek(0)
                image = Image.open(file)
                file.seek(0)

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert to numpy array for quality assessment
            image_array = np.array(image)

            # Assess quality
            quality_metrics = self.assess_image_quality(image_array)

            # Check if image meets quality requirements
            is_acceptable = quality_metrics.overall_score >= self.config.min_quality_score

            # Generate recommendations
            recommendations = self.generate_recommendations(quality_metrics, metadata)

            # Apply preprocessing if image is acceptable
            applied_preprocessing = []
            processed_image = image

            if is_acceptable:
                processed_image, applied_preprocessing = self.preprocess_image(image, quality_metrics)
            else:
                recommendations.append("Image quality is below minimum requirements. Please retake the photo.")

            # Save processed version
            processed_file_path = self.save_temp_image(processed_image, "processed_")

            processing_time_ms = int((time.time() - start_time) * 1000)

            return ImageAnalysisResult(
                metadata=metadata,
                quality_metrics=quality_metrics,
                is_acceptable=is_acceptable,
                processing_recommendations=recommendations,
                preprocessing_applied=applied_preprocessing
            )

        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            raise ValueError(f"Image analysis failed: {str(e)}")

    def clean_temp_files(self, max_age_hours: int = 24):
        """Clean up temporary files older than specified hours"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            for filepath in self.temp_dir.glob("*.jpg"):
                file_age = current_time - filepath.stat().st_mtime
                if file_age > max_age_seconds:
                    filepath.unlink()
                    logger.debug(f"Cleaned up temporary file: {filepath}")

        except Exception as e:
            logger.warning(f"Failed to clean temporary files: {str(e)}")

    async def batch_analyze_images(self, files: List[Union[bytes, io.BytesIO]]) -> List[ImageAnalysisResult]:
        """Analyze multiple images in batch"""
        results = []
        for file in files:
            try:
                result = await self.analyze_image(file)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch image analysis failed for one file: {str(e)}")
                # Create error result
                error_result = ImageAnalysisResult(
                    metadata=None,
                    quality_metrics=None,
                    is_acceptable=False,
                    processing_recommendations=[f"Analysis failed: {str(e)}"],
                    preprocessing_applied=[]
                )
                results.append(error_result)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get image processor statistics"""
        try:
            temp_files_count = len(list(self.temp_dir.glob("*.jpg")))
            total_temp_size = sum(f.stat().st_size for f in self.temp_dir.glob("*.jpg"))

            return {
                "temp_files_count": temp_files_count,
                "total_temp_size_mb": round(total_temp_size / (1024 * 1024), 2),
                "temp_directory": str(self.temp_dir),
                "supported_formats": self.config.supported_formats,
                "max_file_size_mb": self.config.max_file_size_mb,
                "min_quality_score": self.config.min_quality_score,
                "min_dimension": self.config.min_dimension,
                "max_dimension": self.config.max_dimension
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {}