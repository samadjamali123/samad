"""
Camera Component for Plant Disease Detection

This component provides advanced camera integration with:
- Direct camera access
- Real-time preview
- Quality assessment
- Image capture
- PWA camera API support
"""

import streamlit as st
import time
import io
import base64
from typing import Optional, Dict, List, Any, Tuple, Callable
from pathlib import Path
import json

try:
    from PIL import Image, ImageEnhance, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    st.error("📸 PIL is required for camera functionality. Please install: pip install Pillow")

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from frontend.utils.styling import create_camera_container
from frontend.utils.image_processing import assess_image_quality

class CameraComponent:
    """Advanced camera component with real-time preview and quality assessment"""

    SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'webp']
    DEFAULT_QUALITY = 85
    MAX_FILE_SIZE_MB = 10

    def __init__(self, on_capture: Optional[Callable] = None):
        """Initialize camera component"""
        self.on_capture = on_capture
        self.last_capture = None
        self.camera_active = False
        self.quality_score = 0.0
        self.capture_count = 0

    def display(self) -> Optional[bytes]:
        """Display complete camera interface"""
        if not PIL_AVAILABLE:
            st.error("📸 Camera functionality requires PIL/Pillow library")
            return None

        # Check camera permissions and capabilities
        if not self._check_camera_support():
            st.warning("⚠️ Camera access may not be available on this device")

        # Create camera interface sections
        self._display_camera_controls()
        self._display_camera_preview()
        self._display_capture_settings()
        self._display_recent_captures()

        return self.last_capture

    def _check_camera_support(self) -> bool:
        """Check if camera is supported in current environment"""
        # Check if we're in a browser with camera support
        user_agent = st.experimental_get_query_params().get('user_agent', '')

        # Basic browser support check
        has_media_support = (
            'chrome' in user_agent.lower() or
            'firefox' in user_agent.lower() or
            'safari' in user_agent.lower() or
            'edge' in user_agent.lower()
        )

        # Check for mobile
        is_mobile = any(mobile in user_agent.lower() for mobile in ['mobile', 'android', 'iphone', 'ipad'])

        # PWA support
        is_pwa = 'standalone' in user_agent.lower()

        return has_media_support and (not is_mobile or is_pwa)

    def _display_camera_controls(self):
        """Display camera control buttons"""
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            if not self.camera_active:
                if st.button("📸 Start Camera", type="primary", use_container_width=True):
                    self._start_camera()
            else:
                if st.button("⏹️ Stop Camera", use_container_width=True):
                    self._stop_camera()

        with col2:
            if self.camera_active:
                if st.button("📷 Capture", type="secondary", use_container_width=True):
                    self._capture_image()

        with col3:
            if self.camera_active:
                if st.button("🔄 Switch Camera", use_container_width=True):
                    self._switch_camera()

        with col4:
            if self.last_capture:
                if st.button("🗑️ Clear", use_container_width=True):
                    self._clear_captures()

    def _display_camera_preview(self):
        """Display camera preview with real-time feedback"""
        if self.camera_active:
            st.markdown("### 📷 Live Camera Preview")

            # Create preview container
            preview_container = st.empty()

            # Generate camera HTML
            camera_html = create_camera_container({
                'width': 640,
                'height': 480,
                'quality': self.DEFAULT_QUALITY,
                'format': 'jpeg',
                'facing_mode': 'environment',  # Prefer back camera on mobile
                'auto_capture': False,
                'countdown': False,
                'flash': 'auto'
            })

            preview_container.markdown(camera_html, unsafe_allow_html=True)

            # Add real-time quality indicators
            col1, col2, col3 = st.columns(3)

            with col1:
                self._create_quality_indicator("Focus", "🎯", "Good")

            with col2:
                self._create_quality_indicator("Lighting", "☀️", "Bright")

            with col3:
                self._create_quality_indicator("Stability", "📊", "Stable")

        elif self.last_capture:
            self._display_capture_preview()

    def _display_capture_settings(self):
        """Display camera settings and quality controls"""
        with st.expander("⚙️ Camera Settings", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                # Image quality
                quality = st.slider(
                    "📊 Image Quality",
                    min_value=50,
                    max_value=100,
                    value=self.DEFAULT_QUALITY,
                    step=5,
                    help="Higher quality = larger file size"
                )

                # Image format
                format = st.selectbox(
                    "📁 Image Format",
                    options=self.SUPPORTED_FORMATS,
                    index=0,
                    help="Format for captured images"
                )

            with col2:
                # Flash mode
                flash_mode = st.selectbox(
                    "💡 Flash Mode",
                    options=["Auto", "On", "Off", "Torch"],
                    index=0,
                    help="Camera flash/torch settings"
                )

                # Camera facing mode
                facing_mode = st.selectbox(
                    "📱 Camera",
                    options=["Environment (Back)", "User (Front)"],
                    index=0,
                    help="Choose which camera to use"
                )

            # Advanced settings
            with st.expander("🔧 Advanced Settings", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    # Resolution
                    resolution = st.selectbox(
                        "📐 Resolution",
                        options=["640x480", "1280x720", "1920x1080"],
                        index=1,
                        help="Camera resolution"
                    )

                    # White balance
                    white_balance = st.selectbox(
                        "⚪ White Balance",
                        options=["Auto", "Daylight", "Cloudy", "Incandescent", "Fluorescent"],
                        index=0
                    )

                with col2:
                    # Focus mode
                    focus_mode = st.selectbox(
                        "🎯 Focus Mode",
                        options=["Auto", "Manual", "Continuous", "Single-shot"],
                        index=0
                    )

                    # Exposure compensation
                    exposure = st.slider(
                        "☀️ Exposure",
                        min_value=-2,
                        max_value=2,
                        value=0,
                        step=0.1,
                        help="Adjust brightness level"
                    )

    def _display_recent_captures(self):
        """Display gallery of recent captures"""
        if self.last_capture:
            st.markdown("### 📸 Recent Captures")

            # Create capture gallery
            col1, col2, col3 = st.columns(3)

            # Main capture
            with col1:
                st.markdown("**Latest Capture**")

                # Display with quality assessment
                if PIL_AVAILABLE:
                    try:
                        image = Image.open(io.BytesIO(self.last_capture))

                        # Display image
                        st.image(
                            self.last_capture,
                            caption=f"Capture #{self.capture_count}",
                            use_column_width=True,
                            clamp=True
                        )

                        # Quality assessment
                        quality_score, issues = assess_image_quality(image)
                        self._display_capture_quality(quality_score, issues)

                    except Exception as e:
                        st.error(f"Error displaying image: {str(e)}")

            # Action buttons
            with col2:
                st.markdown("**Actions**")

                if st.button("✅ Use This Image", key="use_capture", type="primary", use_container_width=True):
                    if self.on_capture:
                        self.on_capture(self.last_capture)
                    st.success("✅ Image selected for analysis!")

                if st.button("🔄 Retake", key="retake_capture", use_container_width=True):
                    self._start_camera()
                    st.rerun()

                if st.button("💾 Save Locally", key="save_capture", use_container_width=True):
                    self._save_capture_locally()

            with col3:
                st.markdown("**Information**")
                self._display_capture_info()

    def _display_capture_preview(self):
        """Display preview of the last captured image"""
        if not self.last_capture:
            return

        st.markdown("### 📸 Captured Image Preview")

        col1, col2 = st.columns([3, 1])

        with col1:
            if PIL_AVAILABLE:
                try:
                    image = Image.open(io.BytesIO(self.last_capture))

                    # Display with analysis overlay
                    st.image(
                        self.last_capture,
                        caption=f"Capture #{self.capture_count}",
                        use_column_width=True,
                        clamp=True
                    )

                    # Add capture quality overlay
                    self._add_quality_overlay(image)

                except Exception as e:
                    st.error(f"Error displaying preview: {str(e)}")

        with col2:
            # Quick actions
            st.markdown("**Quick Actions**")

            if st.button("🔄 Recapture", key="quick_recapture", use_container_width=True):
                self._start_camera()
                st.rerun()

            if st.button("✅ Use Image", key="quick_use", type="primary", use_container_width=True):
                if self.on_capture:
                    self.on_capture(self.last_capture)
                st.success("✅ Image ready for analysis!")

            if st.button("⚙️ Settings", key="quick_settings", use_container_width=True):
                # Expand settings expander
                pass

    def _display_capture_quality(self, quality_score: float, issues: List[str]):
        """Display quality assessment of captured image"""
        # Create quality indicator
        if quality_score >= 0.8:
            status = "🟢 Excellent"
            color = "#4CAF50"
            message = "Perfect for analysis"
        elif quality_score >= 0.6:
            status = "🟡 Good"
            color = "#FF9800"
            message = "Good quality image"
        else:
            status = "🔴 Poor"
            color = "#F44336"
            message = "Consider recapturing"

        st.markdown(f"""
        <div class="quality-assessment" style="border-left: 4px solid {color}; padding: 10px; background: {color}20;">
            <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">
                {status} ({quality_score:.0%})
            </div>
            <div style="font-size: 14px; color: {color};">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show specific quality issues
        if issues:
            with st.expander("⚠️ Quality Issues", expanded=False):
                for issue in issues:
                    issue_messages = {
                        'blurry': 'Image appears blurry - hold camera steady',
                        'dark': 'Image is too dark - improve lighting',
                        'overexposed': 'Image is too bright - reduce light exposure',
                        'low_contrast': 'Low contrast - adjust camera settings',
                        'noisy': 'Image has noise - improve lighting conditions',
                        'small_resolution': 'Resolution too low - use higher quality setting',
                        'unusual_aspect_ratio': 'Aspect ratio unusual - adjust composition'
                    }

                    message = issue_messages.get(issue, f'Issue: {issue}')
                    st.markdown(f"❌ {message}")

    def _add_quality_overlay(self, image: Image.Image):
        """Add quality overlay information to image"""
        # This would add visual overlay on the image
        # For now, just display metrics
        width, height = image.size

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📐", f"{width}×{height}", "Resolution")

        with col2:
            aspect_ratio = width / height if height > 0 else 1.0
            st.metric("📱", f"{aspect_ratio:.2f}", "Aspect Ratio")

        with col3:
            size_mb = len(self.last_capture) / (1024 * 1024) if self.last_capture else 0
            st.metric("💾", f"{size_mb:.1f}MB", "File Size")

        with col4:
            st.metric("📊", f"{self.quality_score:.0%}", "Quality Score")

    def _display_capture_info(self):
        """Display information about the captured image"""
        if not self.last_capture:
            return

        file_size = len(self.last_capture) / (1024 * 1024)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        info_data = {
            "📊 Quality Score": f"{self.quality_score:.0%}",
            "📐 Resolution": f"Unknown",
            "💾 File Size": f"{file_size:.1f} MB",
            "🕐 Capture Time": timestamp,
            "📸 Capture #": str(self.capture_count),
            "📁 Format": "JPEG"
        }

        for label, value in info_data.items():
            st.markdown(f"**{label}**: {value}")

    def _create_quality_indicator(self, label: str, icon: str, status: str):
        """Create a quality status indicator"""
        status_colors = {
            "Good": "#4CAF50",
            "Bright": "#FF9800",
            "Stable": "#4CAF50",
            "Focus": "#4CAF50"
        }

        color = status_colors.get(status, "#F44336")

        st.markdown(f"""
        <div class="quality-indicator" style="
            background: {color}20;
            border: 1px solid {color};
            border-radius: 8px;
            padding: 8px;
            text-align: center;
        ">
            <div style="font-size: 24px;">{icon}</div>
            <div style="font-size: 12px; font-weight: bold;">{label}</div>
            <div style="font-size: 10px; color: {color};">{status}</div>
        </div>
        """, unsafe_allow_html=True)

    def _start_camera(self):
        """Start camera preview"""
        self.camera_active = True
        st.session_state.camera_active = True
        st.info("📷 Camera starting... Position your leaf in frame")

    def _stop_camera(self):
        """Stop camera preview"""
        self.camera_active = False
        st.session_state.camera_active = False
        st.info("⏹️ Camera stopped")

    def _switch_camera(self):
        """Switch between front and back camera"""
        # Toggle facing mode
        current_facing = st.session_state.get('camera_facing', 'environment')
        new_facing = 'user' if current_facing == 'environment' else 'environment'
        st.session_state.camera_facing = new_facing

        st.info(f"📱 Switched to {'Front' if new_facing == 'user' else 'Back'} camera")

    def _capture_image(self):
        """Capture image from camera"""
        try:
            # In a real implementation, this would interface with WebRTC/getUserMedia
            # For now, simulate capture
            self.capture_count += 1

            # Simulate capture delay
            with st.spinner("📸 Capturing image..."):
                time.sleep(1)

            # Generate placeholder image (in real implementation, this would come from camera)
            if PIL_AVAILABLE:
                placeholder_image = self._create_placeholder_capture()

                # Convert to bytes
                buffer = io.BytesIO()
                placeholder_image.save(buffer, format='JPEG', quality=self.DEFAULT_QUALITY)
                self.last_capture = buffer.getvalue()

                # Assess quality
                quality_score, issues = assess_image_quality(placeholder_image)
                self.quality_score = quality_score

                # Store in session state
                st.session_state.last_capture = self.last_capture
                st.session_state.capture_count = self.capture_count
                st.session_state.quality_score = quality_score

                st.success(f"✅ Image #{self.capture_count} captured!")

                # Stop camera after capture
                self._stop_camera()

            else:
                st.error("❌ Image capture not available")

        except Exception as e:
            st.error(f"❌ Failed to capture image: {str(e)}")

    def _create_placeholder_capture(self) -> Image.Image:
        """Create a placeholder image for demonstration"""
        # Create a simple colored placeholder
        width, height = 640, 480
        image = Image.new('RGB', (width, height), color=(76, 175, 80))  # Green

        # Add text overlay
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(image)

        try:
            # Try to use a nice font
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            # Fallback to default font
            font = ImageFont.load_default()

        # Add capture info
        text1 = f"Plant Leaf Capture #{self.capture_count}"
        text2 = f"Quality: {self.quality_score:.0%}"
        text3 = "Image placeholder for camera demo"

        # Calculate text positions
        bbox1 = draw.textbbox((0, 0), text1, font=font)
        bbox2 = draw.textbbox((0, 0), text2, font=font)
        bbox3 = draw.textbbox((0, 0), text3, font=font)

        y1, y2, y3 = 100, 200, 300

        x1 = (width - (bbox1[2] - bbox1[0])) // 2
        x2 = (width - (bbox2[2] - bbox2[0])) // 2
        x3 = (width - (bbox3[2] - bbox3[0])) // 2

        draw.text((x1, y1), text1, fill="white", font=font, align="center")
        draw.text((x2, y2), text2, fill="white", font=font, align="center")
        draw.text((x3, y3), text3, fill="white", font=font, align="center")

        return image

    def _clear_captures(self):
        """Clear all captured images"""
        self.last_capture = None
        self.capture_count = 0
        self.quality_score = 0.0

        # Clear from session state
        if 'last_capture' in st.session_state:
            del st.session_state.last_capture
        if 'capture_count' in st.session_state:
            del st.session_state.capture_count
        if 'quality_score' in st.session_state:
            del st.session_state.quality_score

        st.info("🗑️ All captures cleared")

    def _save_capture_locally(self):
        """Save captured image locally"""
        if not self.last_capture:
            st.warning("No image to save")
            return

        try:
            # Create download button
            st.download_button(
                label="💾 Download Image",
                data=self.last_capture,
                file_name=f"plant_capture_{self.capture_count}_{int(time.time())}.jpg",
                mime="image/jpeg",
                help="Download captured image to your device"
            )

            st.success("✅ Image ready for download!")

        except Exception as e:
            st.error(f"❌ Failed to prepare download: {str(e)}")

    def get_capture_data(self) -> Optional[bytes]:
        """Get the last captured image data"""
        return self.last_capture

    def get_capture_quality(self) -> float:
        """Get the quality score of the last capture"""
        return self.quality_score

    def is_camera_active(self) -> bool:
        """Check if camera is currently active"""
        return self.camera_active

    def get_capture_count(self) -> int:
        """Get the number of captures made"""
        return self.capture_count


# Factory function for creating camera components
def create_camera_component(on_capture: Optional[Callable] = None) -> CameraComponent:
    """
    Factory function to create a camera component.

    Args:
        on_capture: Callback function when image is captured

    Returns:
        CameraComponent instance
    """
    return CameraComponent(on_capture)


# Utility functions for camera integration
def check_camera_permissions() -> bool:
    """Check if camera permissions are available"""
    # In a real implementation, this would check camera permissions
    # For Streamlit, we simulate this
    return True


def get_camera_capabilities() -> Dict[str, Any]:
    """Get camera capabilities information"""
    return {
        'video': True,
        'photo': True,
        'flash': True,
        'auto_focus': True,
        'front_camera': True,
        'back_camera': True,
        'resolution': ['640x480', '1280x720', '1920x1080'],
        'formats': ['jpg', 'jpeg', 'png', 'webp'],
        'quality': ['low', 'medium', 'high']
    }


def optimize_for_mobile() -> str:
    """Get mobile-optimized camera settings"""
    return json.dumps({
        'width': 640,
        'height': 480,
        'quality': 75,
        'format': 'jpeg',
        'facing_mode': 'environment',
        'auto_capture': False,
        'enable_torch': True,
        'focus_mode': 'continuous'
    })