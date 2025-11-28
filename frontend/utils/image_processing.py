"""
Frontend Image Processing Utilities

This module provides image validation, quality assessment, and preprocessing
functions for the Streamlit frontend interface.
"""

import io
import base64
from typing import Tuple, List, Optional, Any, Dict
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import cv2
from skimage import exposure, filters, measure, morphology

logger = None


def setup_logger():
    """Setup image processing logger"""
    global logger
    import logging
    logger = logging.getLogger(__name__)


# Initialize logger when module is imported
setup_logger()


class ImageValidationError(Exception):
    """Custom exception for image validation errors"""
    pass


class ImageQualityAssessment:
    """Image quality assessment results"""

    def __init__(self, overall_score: float, metrics: Dict[str, Any]):
        self.overall_score = overall_score
        self.metrics = metrics

        # Quality categories
        self.sharpness = metrics.get('sharpness', 0.5)
        self.brightness = metrics.get('brightness', 0.5)
        self.contrast = metrics.get('contrast', 0.5)
        self.noise_level = metrics.get('noise_level', 0.5)
        self.color_balance = metrics.get('color_balance', 0.5)
        self.composition = metrics.get('composition', 0.5)
        self.leaf_coverage = metrics.get('leaf_coverage', 0.5)

        # Issues list
        self.issues = []

        # Categorize quality
        if overall_score < 0.3:
            self.quality_category = "poor"
            self.color = "#F44336"
        elif overall_score < 0.5:
            self.quality_category = "fair"
            self.color = "#FF9800"
        elif overall_score < 0.7:
            self.quality_category = "good"
            self.color = "#FF9800"
        else:
            self.quality_category = "excellent"
            self.color = "#4CAF50"

    def add_issue(self, issue: str):
        """Add quality issue"""
        if issue not in self.issues:
            self.issues.append(issue)

    def get_recommendations(self) -> List[str]:
        """Get improvement recommendations based on quality"""
        recommendations = []

        if self.sharpness < 0.4:
            recommendations.append("Image appears blurry. Hold camera steady or use a tripod.")

        if self.brightness < 0.4:
            recommendations.append("Image is too dark. Use better lighting or adjust exposure.")
        elif self.brightness > 0.7:
            recommendations.append("Image is too bright. Reduce brightness or use diffused lighting.")

        if self.contrast < 0.4:
            recommendations.append("Low contrast between leaf and background. Use more even lighting.")

        if self.noise_level > 0.6:
            recommendations.append("High digital noise. Use lower ISO or better lighting.")

        if self.color_balance < 0.4:
            recommendations.append("Color balance seems off. Check white balance settings.")

        if self.composition < 0.4:
            recommendations.append("Leaf doesn't fill frame properly. Get closer to the subject.")

        if self.leaf_coverage < 0.3:
            recommendations.append("Leaf coverage is too low. Focus more on the affected leaf area.")

        return recommendations


def validate_image_format(file_path: str, file_data: Optional[bytes] = None) -> Tuple[bool, str]:
    """
    Validate image file format

    Args:
        file_path: Path to the file or filename
        file_data: Optional file data for validation

    Returns:
        Tuple of (is_valid, format_name)
    """
    try:
        # Check file extension
        file_ext = Path(file_path).suffix.lower()

        supported_formats = {'.jpg', '.jpeg', '.png', '.webp'}

        if file_ext not in supported_formats:
            return False, f"Unsupported format: {file_ext}"

        # Validate with PIL if file data is available
        if file_data:
            try:
                image = Image.open(io.BytesIO(file_data))
                detected_format = image.format.lower() if image.format else 'unknown'

                if detected_format not in ['jpeg', 'png', 'webp']:
                    return False, f"Invalid image format: {detected_format}"

                return True, detected_format

            except Exception as e:
                logger.warning(f"PIL format validation failed: {str(e)}")
                return False, f"Cannot read image file: {str(e)}"

        return True, file_ext.lstrip('.')

    except Exception as e:
        logger.error(f"Format validation error: {str(e)}")
        return False, f"Validation error: {str(e)}"


def validate_image_size(file_data: bytes, max_size_mb: float = 10.0, min_width: int = 512, min_height: int = 512) -> Tuple[bool, float]:
    """
    Validate image file size and dimensions

    Args:
        file_data: Image file data
        max_size_mb: Maximum allowed size in MB
        min_width: Minimum allowed width in pixels
        min_height: Minimum allowed height in pixels

    Returns:
        Tuple of (is_valid, size_mb)
    """
    try:
        # Check file size
        file_size_mb = len(file_data) / (1024 * 1024)

        if file_size_mb > max_size_mb:
            return False, file_size_mb

        # Check image dimensions
        try:
            image = Image.open(io.BytesIO(file_data))
            width, height = image.size

            if width < min_width or height < min_height:
                logger.warning(f"Image dimensions too small: {width}x{height}")
                return False, file_size_mb

        except Exception as e:
            logger.error(f"Dimension validation failed: {str(e)}")
            return False, file_size_mb

        return True, file_size_mb

    except Exception as e:
        logger.error(f"Size validation error: {str(e)}")
        return False, 0.0


def assess_image_quality(image_data: bytes) -> ImageQualityAssessment:
    """
    Assess image quality using multiple metrics

    Args:
        image_data: Image file data as bytes

    Returns:
        ImageQualityAssessment object with quality metrics
    """
    try:
        # Open image and convert to numpy array
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')  # Ensure RGB format
        img_array = np.array(image)

        metrics = {}

        # 1. Sharpness assessment using Laplacian variance
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 100.0)  # Normalize to 0-1
        metrics['sharpness'] = sharpness_score

        # 2. Brightness assessment
        mean_brightness = np.mean(gray)
        # Optimal brightness is around 128 (middle of 0-255 range)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
        metrics['brightness'] = brightness_score

        # 3. Contrast assessment using standard deviation
        contrast_score = min(1.0, np.std(gray) / 64.0)  # Normalize to 0-1
        metrics['contrast'] = contrast_score

        # 4. Noise assessment using median filter difference
        median_filtered = cv2.medianBlur(gray, 3)
        noise_level = np.mean(cv2.absdiff(gray, median_filtered))
        noise_score = 1.0 - min(1.0, noise_level / 50.0)  # Normalize and invert
        metrics['noise_level'] = noise_score

        # 5. Color balance assessment
        if len(img_array.shape) == 3:
            r_mean = np.mean(img_array[:, :, 0])
            g_mean = np.mean(img_array[:, :, 1])
            b_mean = np.mean(img_array[:, :, 2])

            # Calculate color balance (how close RGB means are to each other)
            color_mean = np.mean([r_mean, g_mean, b_mean])
            color_balance_score = 1.0 - (max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)) / 255.0
        else:
            color_balance_score = 0.8  # Default for grayscale
        metrics['color_balance'] = color_balance_score

        # 6. Composition assessment - focus on central area
        height, width = gray.shape
        center_x, center_y = width // 2, height // 2
        center_region = gray[center_y-height//4:center_y+height//4,
                            center_x-width//4:center_x+width//4]
        edge_region = np.concatenate([
            gray[:height//4, :width//4],  # Top-left
            gray[:height//4, -width//4:],  # Top-right
            gray[-height//4:, :width//4],  # Bottom-left
            gray[-height//4:, -width//4:]   # Bottom-right
        ])

        center_mean = np.mean(center_region)
        edge_mean = np.mean(edge_region)

        composition_score = min(1.0, max(0.0, (center_mean - edge_mean) / 64.0 + 0.5))
        metrics['composition'] = composition_score

        # 7. Leaf coverage estimation using simple thresholding
        try:
            # Use Otsu's method for automatic thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            leaf_pixels = np.sum(binary > 0)
            total_pixels = binary.size
            leaf_coverage = leaf_pixels / total_pixels

            # Score based on coverage (optimal around 30-70%)
            if leaf_coverage < 0.2:
                coverage_score = leaf_coverage * 5  # Scale up for very low coverage
            elif leaf_coverage > 0.8:
                coverage_score = (1.0 - leaf_coverage) * 5  # Penalty for too much coverage
            else:
                coverage_score = min(1.0, leaf_coverage * 1.5)

            metrics['leaf_coverage'] = coverage_score

        except Exception as e:
            logger.warning(f"Leaf coverage assessment failed: {str(e)}")
            metrics['leaf_coverage'] = 0.5

        # Calculate overall score (weighted average)
        weights = {
            'sharpness': 0.2,
            'brightness': 0.15,
            'contrast': 0.15,
            'noise_level': 0.15,
            'color_balance': 0.1,
            'composition': 0.15,
            'leaf_coverage': 0.1
        }

        overall_score = sum(metrics[metric] * weight for metric, weight in weights.items())

        # Create assessment object
        assessment = ImageQualityAssessment(overall_score, metrics)

        # Add issues based on thresholds
        if sharpness_score < 0.4:
            assessment.add_issue('blurry')
        if brightness_score < 0.4:
            assessment.add_issue('dark')
        elif brightness_score > 0.7:
            assessment.add_issue('overexposed')
        if contrast_score < 0.4:
            assessment.add_issue('low_contrast')
        if noise_score < 0.4:
            assessment.add_issue('noisy')
        if color_balance_score < 0.4:
            assessment.add_issue('color_cast')
        if composition_score < 0.4:
            assessment.add_issue('poor_composition')
        if leaf_coverage < 0.3:
            assessment.add_issue('small_resolution')

        return assessment

    except Exception as e:
        logger.error(f"Quality assessment failed: {str(e)}")
        # Return default assessment on error
        return ImageQualityAssessment(0.5, {})


def resize_for_display(image: Image.Image, max_size: int = 800, quality: int = 85) -> Image.Image:
    """
    Resize image for web display while maintaining aspect ratio

    Args:
        image: PIL Image object
        max_size: Maximum dimension for display
        quality: JPEG quality (0-100)

    Returns:
        Resized PIL Image object
    """
    try:
        # Get original dimensions
        original_width, original_height = image.size

        # Calculate new dimensions maintaining aspect ratio
        if max(original_width, original_height) <= max_size:
            return image  # No resize needed

        if original_width > original_height:
            # Landscape orientation
            new_width = max_size
            new_height = int(max_size * original_height / original_width)
        else:
            # Portrait orientation
            new_height = max_size
            new_width = int(max_size * original_width / original_height)

        # Resize with high-quality resampling
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return resized_image

    except Exception as e:
        logger.error(f"Image resize failed: {str(e)}")
        return image


def create_thumbnail(image: Image.Image, size: Tuple[int, int] = (150, 150)) -> Image.Image:
    """
    Create thumbnail of the image

    Args:
        image: PIL Image object
        size: Tuple of (width, height) for thumbnail

    Returns:
        Thumbnail PIL Image object
    """
    try:
        # Create thumbnail with anti-aliasing
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)

        return thumbnail

    except Exception as e:
        logger.error(f"Thumbnail creation failed: {str(e)}")
        return image


def enhance_image_for_analysis(image: Image.Image) -> Image.Image:
    """
    Enhance image for better AI analysis

    Args:
        image: PIL Image object

    Returns:
        Enhanced PIL Image object
    """
    try:
        enhanced = image.copy()

        # Convert to RGB if necessary
        if enhanced.mode != 'RGB':
            enhanced = enhanced.convert('RGB')

        # Apply slight sharpening
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.1)

        # Apply slight contrast enhancement
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.1)

        # Apply slight color enhancement
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(1.05)

        return enhanced

    except Exception as e:
        logger.error(f"Image enhancement failed: {str(e)}")
        return image


def crop_to_leaf_area(image: Image.Image, confidence: float = 0.7) -> Image.Image:
    """
    Automatically crop to the most likely leaf area using simple heuristics

    Args:
        image: PIL Image object
        confidence: Confidence threshold for cropping

    Returns:
        Cropped PIL Image object
    """
    try:
        if confidence < 0.6:
            # Low confidence - use original image
            return image

        # Convert to grayscale for processing
        gray = image.convert('L')
        img_array = np.array(gray)

        # Apply threshold to find main subject
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # Find largest contour (likely the leaf)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add padding (10% of dimensions)
        padding_x = int(w * 0.05)
        padding_y = int(h * 0.05)

        # Ensure bounding box is within image bounds
        x_start = max(0, x - padding_x)
        y_start = max(0, y - padding_y)
        x_end = min(image.width, x + w + padding_x)
        y_end = min(image.height, y + h + padding_y)

        # Crop the image
        cropped = image.crop((x_start, y_start, x_end, y_end))

        return cropped

    except Exception as e:
        logger.error(f"Leaf area cropping failed: {str(e)}")
        return image


def optimize_for_web(image: Image.Image, format: str = 'JPEG', quality: int = 85) -> bytes:
    """
    Optimize image for web delivery

    Args:
        image: PIL Image object
        format: Output format (JPEG, PNG, WEBP)
        quality: Compression quality (0-100)

    Returns:
        Optimized image as bytes
    """
    try:
        # Ensure RGB format for web compatibility
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Save to bytes with optimization
        buffer = io.BytesIO()

        if format.upper() == 'JPEG':
            image.save(buffer, format='JPEG', quality=quality, optimize=True, progressive=True)
        elif format.upper() == 'PNG':
            image.save(buffer, format='PNG', optimize=True, compress_level=6)
        elif format.upper() == 'WEBP':
            image.save(buffer, format='WEBP', quality=quality, optimize=True, method=6)
        else:
            # Default to JPEG
            image.save(buffer, format='JPEG', quality=quality, optimize=True)

        return buffer.getvalue()

    except Exception as e:
        logger.error(f"Web optimization failed: {str(e)}")
        # Return original image bytes on error
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        return buffer.getvalue()


def extract_exif_data(image_data: bytes) -> Dict[str, Any]:
    """
    Extract EXIF metadata from image

    Args:
        image_data: Image file data as bytes

    Returns:
        Dictionary with EXIF data
    """
    try:
        from PIL.ExifTags import TAGS

        image = Image.open(io.BytesIO(image_data))
        exif_data = image._getexif()

        if exif_data is None:
            return {}

        # Extract common EXIF tags
        extracted_data = {}

        for tag, value in exif_data.items():
            if tag in TAGS:
                tag_name = TAGS[tag]
                extracted_data[tag_name] = value

        return extracted_data

    except Exception as e:
        logger.warning(f"EXIF extraction failed: {str(e)}")
        return {}


def create_image_analysis_report(assessment: ImageQualityAssessment) -> Dict[str, Any]:
    """
    Create a comprehensive image analysis report

    Args:
        assessment: ImageQualityAssessment object

    Returns:
        Dictionary with analysis report
    """
    return {
        'overall_score': assessment.overall_score,
        'quality_category': assessment.quality_category,
        'metrics': {
            'sharpness': assessment.sharpness,
            'brightness': assessment.brightness,
            'contrast': assessment.contrast,
            'noise_level': assessment.noise_level,
            'color_balance': assessment.color_balance,
            'composition': assessment.composition,
            'leaf_coverage': assessment.leaf_coverage
        },
        'issues': assessment.issues,
        'recommendations': assessment.get_recommendations(),
        'color_code': assessment.color,
        'is_acceptable': assessment.overall_score >= 0.4
    }


def batch_process_images(image_data_list: List[bytes]) -> List[Dict[str, Any]]:
    """
    Process multiple images in batch

    Args:
        image_data_list: List of image file data

    Returns:
        List of analysis reports for each image
    """
    results = []

    for i, image_data in enumerate(image_data_list):
        try:
            # Validate each image
            format_valid, format_name = validate_image_format(f"image_{i}.jpg", image_data)
            size_valid, size_mb = validate_image_size(image_data)

            if not format_valid:
                results.append({
                    'index': i,
                    'success': False,
                    'error': f"Invalid format: {format_name}",
                    'size_mb': size_mb
                })
                continue

            if not size_valid:
                results.append({
                    'index': i,
                    'success': False,
                    'error': f"File too large: {size_mb:.2f}MB",
                    'format': format_name
                })
                continue

            # Assess quality
            assessment = assess_image_quality(image_data)
            report = create_image_analysis_report(assessment)
            report.update({
                'index': i,
                'success': True,
                'format': format_name,
                'size_mb': size_mb
            })

            results.append(report)

        except Exception as e:
            logger.error(f"Batch processing error for image {i}: {str(e)}")
            results.append({
                'index': i,
                'success': False,
                'error': f"Processing error: {str(e)}"
            })

    return results


def validate_compatibility(image_data: bytes) -> Dict[str, Any]:
    """
    Check image compatibility with AI models

    Args:
        image_data: Image file data as bytes

    Returns:
        Dictionary with compatibility information
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        width, height = image.size

        # Basic compatibility checks
        compatibility = {
            'is_valid': True,
            'format': image.format.lower() if image.format else 'unknown',
            'dimensions': (width, height),
            'mode': image.mode,
            'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info,
            'color_depth': len(image.getbands()),
            'min_size_ok': width >= 512 and height >= 512,
            'max_size_ok': width <= 2048 and height <= 2048,
            'aspect_ratio': width / height if height > 0 else 1.0,
            'file_size_mb': len(image_data) / (1024 * 1024)
        }

        # AI model compatibility
        compatibility['ai_ready'] = (
            compatibility['min_size_ok'] and
            compatibility['max_size_ok'] and
            not compatibility['has_transparency'] and
            compatibility['format'] in ['JPEG', 'PNG']
        )

        # Quality warnings for AI
        if width < 512 or height < 512:
            compatibility['quality_warning'] = "Low resolution may affect AI accuracy"
        elif compatibility['file_size_mb'] > 5:
            compatibility['quality_warning'] = "Large file size may slow processing"
        else:
            compatibility['quality_warning'] = None

        return compatibility

    except Exception as e:
        logger.error(f"Compatibility check failed: {str(e)}")
        return {
            'is_valid': False,
            'error': str(e),
            'ai_ready': False
        }


# Utility functions for Streamlit integration
def get_image_info(image_data: bytes) -> Dict[str, Any]:
    """
    Get comprehensive image information for Streamlit display

    Args:
        image_data: Image file data as bytes

    Returns:
        Dictionary with image information
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        width, height = image.size

        return {
            'width': width,
            'height': height,
            'megapixels': round((width * height) / 1000000, 2),
            'aspect_ratio': round(width / height if height > 0 else 1.0, 2),
            'format': image.format.lower() if image.format else 'unknown',
            'mode': image.mode,
            'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info,
            'color_count': len(image.getbands()),
            'file_size_mb': round(len(image_data) / (1024 * 1024), 2),
            'orientation': 'landscape' if width > height else 'portrait' if height > width else 'square'
        }

    except Exception as e:
        logger.error(f"Image info extraction failed: {str(e)}")
        return {
            'error': str(e),
            'width': 0,
            'height': 0,
            'megapixels': 0,
            'aspect_ratio': 0,
            'format': 'unknown',
            'mode': 'unknown'
        }


def create_download_link(image_data: bytes, filename: str = "processed_image.jpg") -> str:
    """
    Create a download link for processed image

    Args:
        image_data: Image file data as bytes
        filename: Filename for download

    Returns:
        Base64 encoded data URL
    """
    try:
        # Encode as base64
        b64_data = base64.b64encode(image_data).decode()

        # Create data URL
        mime_type = "image/jpeg"
        data_url = f"data:{mime_type};base64,{b64_data}"

        return data_url

    except Exception as e:
        logger.error(f"Download link creation failed: {str(e)}")
        return ""