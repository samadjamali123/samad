"""
Image Uploader Component

This component handles image upload, validation, preview, and quality assessment
with drag-and-drop functionality and user-friendly error handling.
"""

import streamlit as st
import io
import time
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import base64
from PIL import Image, ImageEnhance
import numpy as np

from frontend.utils.api_client import APIClient
from frontend.utils.image_processing import validate_image_format, validate_image_size, assess_image_quality

class ImageUploader:
    """Advanced image uploader with validation and quality assessment"""

    SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'webp']
    MAX_FILE_SIZE_MB = 10
    MIN_IMAGE_SIZE = 512
    MAX_IMAGE_SIZE = 1024

    def __init__(self, api_client: APIClient):
        """Initialize uploader with API client"""
        self.api_client = api_client
        self.uploaded_file = None
        self.validation_results = None
        self.last_upload_time = None

    def display(self) -> Optional[bytes]:
        """Display the complete image uploader interface"""
        # Create upload interface sections
        self._display_header()
        self._display_upload_area()
        self._display_tips()
        self._display_validation_results()

        return self.uploaded_file

    def _display_header(self):
        """Display uploader header with status information"""
        st.markdown("""
        <div class="uploader-header">
            <div class="uploader-title">
                <h3>📸 Upload Plant Leaf Image</h3>
                <p>Take a clear photo of a plant leaf showing any symptoms</p>
            </div>
            <div class="uploader-status">
                <div class="status-indicator ready"></div>
                <span>Ready for upload</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def _display_upload_area(self):
        """Create the main drag-and-drop upload area"""
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a leaf image or drag and drop",
            type=self.SUPPORTED_FORMATS,
            key="leaf_image_uploader",
            help=f"Supported formats: {', '.join(self.SUPPORTED_FORMATS).upper()} (Max: {self.MAX_FILE_SIZE_MB}MB)",
            accept_multiple_files=False
        )

        if uploaded_file is not None:
            self.last_upload_time = time.time()
            self._process_uploaded_file(uploaded_file)

    def _process_uploaded_file(self, uploaded_file) -> bool:
        """Process and validate the uploaded file"""
        try:
            # Reset state
            self.uploaded_file = None
            self.validation_results = None

            # Read file data
            file_data = uploaded_file.read()
            if not file_data:
                st.error("❌ The uploaded file is empty. Please try again.")
                return False

            # Validate file format
            format_valid, detected_format = validate_image_format(uploaded_file.name, file_data)
            if not format_valid:
                st.error(f"❌ Invalid file format. Supported formats: {', '.join(self.SUPPORTED_FORMATS).upper()}")
                return False

            # Validate file size
            size_valid, file_size_mb = validate_image_size(file_data)
            if not size_valid:
                st.error(f"❌ File too large. Maximum size: {self.MAX_FILE_SIZE_MB}MB. Your file: {file_size_mb:.1f}MB")
                return False

            # Open image for validation
            try:
                image = Image.open(io.BytesIO(file_data))
                image = image.convert('RGB')  # Convert to RGB for consistency
            except Exception as e:
                st.error(f"❌ Cannot read image file: {str(e)}")
                return False

            # Validate image dimensions
            width, height = image.size
            if width < self.MIN_IMAGE_SIZE or height < self.MIN_IMAGE_SIZE:
                st.error(f"❌ Image too small. Minimum size: {self.MIN_IMAGE_SIZE}x{self.MIN_IMAGE_SIZE} pixels. Your image: {width}x{height}")
                return False

            # Assess image quality
            quality_score, quality_issues = assess_image_quality(image)

            # Store results
            self.uploaded_file = file_data
            self.validation_results = {
                'filename': uploaded_file.name,
                'format': detected_format,
                'size_mb': file_size_mb,
                'dimensions': (width, height),
                'quality_score': quality_score,
                'quality_issues': quality_issues,
                'is_valid': len(quality_issues) == 0,
                'image': image
            }

            # Show success message
            if quality_score >= 0.6:
                st.success(f"✅ Image uploaded successfully! Quality score: {quality_score:.1%}")
            elif quality_score >= 0.4:
                st.warning(f"⚠️ Image uploaded with fair quality: {quality_score:.1%}. Results may be less accurate.")
            else:
                st.error(f"❌ Low image quality: {quality_score:.1%}. For best results, retake the photo.")

            # Display image preview
            self._display_image_preview()

            return True

        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            return False

    def _display_image_preview(self):
        """Display the uploaded image with quality information"""
        if not self.validation_results:
            return

        image = self.validation_results['image']
        filename = self.validation_results['filename']
        quality_score = self.validation_results['quality_score']

        # Create columns for layout
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Display image
            st.markdown("**Image Preview**")
            st.image(
                image,
                caption=filename,
                use_column_width=True,
                clamp=True
            )

        with col2:
            # Display quality metrics
            st.markdown("**Quality Analysis**")

            # Overall score with color coding
            score_color = self._get_quality_color(quality_score)
            st.markdown(f"""
            <div class="quality-score" style="color: {score_color};">
                <div class="score-circle">{quality_score:.0%}</div>
                <div class="score-label">Quality Score</div>
            </div>
            """, unsafe_allow_html=True)

            # Detailed metrics
            self._display_quality_metrics()

        with col3:
            # Display actions
            st.markdown("**Actions**")

            if st.button("🔄 Retake Photo", key="retake_photo", use_container_width=True):
                # Clear current upload
                if 'leaf_image_uploader' in st.session_state:
                    st.session_state.leaf_image_uploader = None
                self.uploaded_file = None
                self.validation_results = None
                st.rerun()

            if st.button("✅ Use This Photo", key="use_photo", type="primary", use_container_width=True):
                if quality_score >= 0.4:
                    st.success("✅ Image accepted for analysis!")
                else:
                    st.warning("⚠️ Using low-quality image may reduce accuracy.")

            # Show file info
            st.markdown("**File Information**")
            st.markdown(f"""
            <div class="file-info">
                <div class="info-item">
                    <span class="info-label">Format:</span>
                    <span class="info-value">{self.validation_results['format'].upper()}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Size:</span>
                    <span class="info-value">{self.validation_results['size_mb']:.1f} MB</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Dimensions:</span>
                    <span class="info-value">{self.validation_results['dimensions'][0]}×{self.validation_results['dimensions'][1]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    def _display_quality_metrics(self):
        """Display detailed quality assessment metrics"""
        if not self.validation_results:
            return

        # Get quality issues
        issues = self.validation_results['quality_issues']
        quality_score = self.validation_results['quality_score']

        # Create quality indicators
        metrics = {
            'Sharpness': '✅' if 'blurry' not in issues else '❌',
            'Lighting': '✅' if 'dark' not in issues and 'overexposed' not in issues else '❌',
            'Contrast': '✅' if 'low_contrast' not in issues else '❌',
            'Composition': '✅' if 'poor_composition' not in issues else '❌',
            'Noise': '✅' if 'noisy' not in issues else '❌'
        }

        for metric, status in metrics.items():
            st.markdown(f"""
            <div class="metric-item">
                <span class="metric-status">{status}</span>
                <span class="metric-name">{metric}</span>
            </div>
            """, unsafe_allow_html=True)

    def _display_tips(self):
        """Display helpful tips for taking good leaf photos"""
        with st.expander("📋 Tips for Best Results", expanded=False):
            tips = [
                {
                    'icon': '☀️',
                    'title': 'Good Lighting',
                    'description': 'Take photos in bright, indirect daylight. Avoid shadows and direct sunlight.'
                },
                {
                    'icon': '📱',
                    'title': 'Camera Focus',
                    'description': 'Ensure the leaf is in sharp focus. Tap to focus on mobile devices.'
                },
                {
                    'icon': '🖼️',
                    'title': 'Fill the Frame',
                    'description': 'Get close enough so the leaf fills at least 50% of the frame.'
                },
                {
                    'icon': '🌿',
                    'title': 'Show Symptoms',
                    'description': 'Include any visible spots, discoloration, or damage in the photo.'
                },
                {
                    'icon': '🔄',
                    'title': 'Multiple Angles',
                    'description': 'Take photos from different angles if symptoms are unclear.'
                },
                {
                    'icon': '🎨',
                    'title': 'Simple Background',
                    'description': 'Use a neutral background. Avoid busy patterns that might confuse the AI.'
                }
            ]

            for tip in tips:
                st.markdown(f"""
                <div class="tip-card">
                    <div class="tip-icon">{tip['icon']}</div>
                    <div class="tip-content">
                        <h4>{tip['title']}</h4>
                        <p>{tip['description']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    def _display_validation_results(self):
        """Display detailed validation results if issues were found"""
        if not self.validation_results or self.validation_results['is_valid']:
            return

        issues = self.validation_results['quality_issues']
        quality_score = self.validation_results['quality_score']

        st.markdown("### ⚠️ Image Quality Recommendations")

        recommendations = []

        if 'blurry' in issues:
            recommendations.append({
                'issue': 'Image appears blurry',
                'fix': 'Keep the camera steady and use good lighting. Consider using a tripod or steady surface.'
            })

        if 'dark' in issues:
            recommendations.append({
                'issue': 'Image is too dark',
                'fix': 'Move to better lighting or use a flash. Aim for even, bright lighting.'
            })

        if 'overexposed' in issues:
            recommendations.append({
                'issue': 'Image is too bright',
                'fix': 'Move away from direct sunlight or reduce exposure. Use diffused lighting.'
            })

        if 'low_contrast' in issues:
            recommendations.append({
                'issue': 'Low contrast between leaf and background',
                'fix': 'Use a contrasting background or adjust camera settings. Ensure good lighting.'
            })

        if 'noisy' in issues:
            recommendations.append({
                'issue': 'Image has digital noise',
                'fix': 'Use better lighting or lower ISO settings. Keep the camera steady.'
            })

        if 'poor_composition' in issues:
            recommendations.append({
                'issue': 'Leaf doesn\'t fill the frame properly',
                'fix': 'Get closer to the leaf or use zoom. Ensure the leaf is the main subject.'
            })

        if 'small_resolution' in issues:
            recommendations.append({
                'issue': 'Image resolution is too low',
                'fix': 'Use a higher resolution camera setting. Minimum required: 512×512 pixels.'
            })

        for i, rec in enumerate(recommendations):
            with st.expander(f"❌ {rec['issue']}", expanded=(len(recommendations) <= 2)):
                st.markdown(f"""
                <div class="recommendation">
                    <div class="recommendation-issue">❌ {rec['issue']}</div>
                    <div class="recommendation-fix">
                        <strong>Fix:</strong> {rec['fix']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Show continue warning
        if quality_score >= 0.4:
            st.warning("""
            ⚠️ **Low Quality Image Detected**

            You can still proceed with analysis, but results may be less accurate.
            For best results, consider retaking the photo following the recommendations above.
            """)
        else:
            st.error("""
            ❌ **Image Quality Too Low**

            The image quality is insufficient for reliable analysis.
            Please retake the photo following the recommendations above.
            """)

    def _get_quality_color(self, score: float) -> str:
        """Get color based on quality score"""
        if score >= 0.7:
            return "#4CAF50"  # Green
        elif score >= 0.4:
            return "#FF9800"  # Orange
        else:
            return "#F44336"  # Red

    def get_upload_stats(self) -> Dict[str, Any]:
        """Get upload statistics for debugging"""
        return {
            'has_upload': self.uploaded_file is not None,
            'upload_time': self.last_upload_time,
            'validation_results': self.validation_results,
            'api_available': self.api_client.health_check() if hasattr(self.api_client, 'health_check') else True
        }

    def reset(self):
        """Reset uploader state"""
        self.uploaded_file = None
        self.validation_results = None
        self.last_upload_time = None

# Helper functions for image validation
def validate_image_format(filename: str, file_data: bytes) -> Tuple[bool, str]:
    """Validate image file format"""
    try:
        # Check file extension
        ext = Path(filename).suffix.lower().lstrip('.')
        if ext not in ImageUploader.SUPPORTED_FORMATS:
            return False, ext

        # Try to open with PIL
        image = Image.open(io.BytesIO(file_data))
        detected_format = image.format.lower() if image.format else 'unknown'

        return detected_format in ImageUploader.SUPPORTED_FORMATS, detected_format

    except Exception:
        return False, 'unknown'

def validate_image_size(file_data: bytes) -> Tuple[bool, float]:
    """Validate image file size"""
    size_mb = len(file_data) / (1024 * 1024)
    is_valid = size_mb <= ImageUploader.MAX_FILE_SIZE_MB
    return is_valid, size_mb

def assess_image_quality(image: Image.Image) -> Tuple[float, List[str]]:
    """Assess image quality and return score with issues"""
    issues = []
    quality_score = 0.8  # Start with good score and deduct for issues

    # Convert to numpy array for analysis
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    # Check resolution
    if max(width, height) < 512:
        issues.append('small_resolution')
        quality_score -= 0.3

    # Check if image is too small relative to max dimension
    if max(width, height) < 800:
        issues.append('low_resolution')
        quality_score -= 0.2

    # Check image dimensions ratio
    aspect_ratio = width / height
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        issues.append('unusual_aspect_ratio')
        quality_score -= 0.1

    # Check for blur (simplified Laplacian variance)
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array

    laplacian_var = np.var(gray)
    if laplacian_var < 100:
        issues.append('blurry')
        quality_score -= 0.25
    elif laplacian_var < 200:
        issues.append('slight_blur')
        quality_score -= 0.1

    # Check brightness
    mean_brightness = np.mean(gray)
    if mean_brightness < 50:
        issues.append('dark')
        quality_score -= 0.2
    elif mean_brightness > 200:
        issues.append('overexposed')
        quality_score -= 0.2

    # Check contrast
    std_brightness = np.std(gray)
    if std_brightness < 30:
        issues.append('low_contrast')
        quality_score -= 0.15

    # Check for noise (simplified)
    if std_brightness > 80:
        issues.append('noisy')
        quality_score -= 0.1

    # Ensure score doesn't go below 0
    quality_score = max(0.0, quality_score)

    return quality_score, issues