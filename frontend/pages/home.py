"""
Home Page - Main Disease Detection Interface
"""

import os
import io
import time
from datetime import datetime
from typing import Dict, Any, Optional
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# Import utilities
from utils.api_client import APIClient
from utils.ui_helpers import (
    show_page_header,
    show_empty_state,
    create_confidence_bar,
    create_metric_card,
    create_disease_card,
    create_severity_badge,
    format_file_size,
    format_processing_time,
    create_confidence_gauge,
    show_error_message,
    show_success_message,
    create_download_button
)


def show_home_page():
    """Display the main disease detection page"""
    show_page_header("🌿 Plant Disease Detection", "Upload plant leaf images for AI-powered analysis")

    # Check API connection
    if 'api_client' not in st.session_state:
        show_empty_state(
            icon="🔌",
            title="API Not Connected",
            description="Please ensure the backend API is running and try again.",
            action_text="Check Connection",
            action_callback=lambda: st.rerun()
        )
        return

    # Main detection interface
    col1, col2 = st.columns([2, 1])

    with col1:
        show_upload_section()

    with col2:
        show_quick_stats()

    # Detection history section
    show_detection_history()


def show_upload_section():
    """Show image upload and detection interface"""
    st.markdown("### 📤 Upload Plant Image")
    st.markdown("Upload a clear image of a plant leaf to detect potential diseases.")

    # Image upload widget
    uploaded_file = st.file_uploader(
        "Choose an image file...",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Supported formats: JPEG, PNG, WebP. Maximum size: 10MB",
        key="image_uploader"
    )

    if uploaded_file:
        # Display uploaded image
        st.markdown("#### Preview")
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"**{uploaded_file.name}** ({format_file_size(uploaded_file.size)})", use_column_width=True)
        except Exception as e:
            show_error_message(f"Error loading image: {str(e)}")
            return

        # Analysis options
        with st.expander("⚙️ Analysis Options", expanded=False):
            # Confidence threshold slider
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.get('confidence_threshold', 0.6),
                step=0.1,
                help="Minimum confidence level for disease detection"
            )
            st.session_state.confidence_threshold = confidence_threshold

            # Advanced options
            col1, col2 = st.columns(2)
            with col1:
                enhance_contrast = st.checkbox("Enhance Contrast", value=True, help="Apply contrast enhancement for better detection")
            with col2:
                include_history = st.checkbox("Save to History", value=True, help="Include this analysis in your detection history")

        # Detection button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Detect Disease", type="primary", use_container_width=True, key="detect_button"):
                perform_disease_detection(
                    uploaded_file,
                    confidence_threshold,
                    enhance_contrast,
                    include_history
                )

    else:
        # Show sample images or instructions
        st.markdown("---")
        st.markdown("#### 📋 Instructions")
        instructions = [
            "1. **Select a clear image** of a plant leaf showing symptoms",
            "2. **Ensure good lighting** - natural daylight works best",
            "3. **Include the entire leaf** or affected area if possible",
            "4. **Avoid blurry images** - focus on the affected areas",
            "5. **Multiple angles** help with accurate detection"
        ]
        for instruction in instructions:
            st.markdown(instruction)

        # Show sample disease images if available
        show_sample_diseases()


def show_sample_diseases():
    """Display sample disease information"""
    st.markdown("#### 🌱 Common Plant Diseases")

    sample_diseases = [
        {
            "name": "Leaf Rust",
            "category": "Fungal",
            "severity": "Moderate",
            "description": "Orange-red pustules on leaf surfaces"
        },
        {
            "name": "Powdery Mildew",
            "category": "Fungal",
            "severity": "Moderate",
            "description": "White, powdery coating on leaf surfaces"
        },
        {
            "name": "Early Blight",
            "category": "Fungal",
            "severity": "High",
            "description": "Dark brown to black concentric rings on leaves"
        }
    ]

    for disease in sample_diseases:
        with st.expander(f"🦠 {disease['name']} ({disease['severity']})", expanded=False):
            st.markdown(f"**Category:** {disease['category']}")
            st.markdown(f"**Severity:** {create_severity_badge(disease['severity'])}")
            st.markdown(f"**Description:** {disease['description']}")
            st.markdown("---")


def show_quick_stats():
    """Show quick statistics sidebar"""
    st.markdown("### 📊 Quick Stats")

    # Get API statistics
    api_client = st.session_state.api_client
    stats_response = api_client.get_database_statistics()

    if stats_response.get("success"):
        stats = stats_response.get("statistics", {})

        # Display metric cards
        create_metric_card(
            "Total Diseases",
            str(stats.get("total_diseases", 0))
        )

        create_metric_card(
            "Categories",
            str(stats.get("total_categories", 0))
        )

        create_metric_card(
            "Affected Plants",
            str(stats.get("total_affected_plants", 0))
        )

        create_metric_card(
            "Contagious",
            str(stats.get("contagious_diseases", 0))
        )
    else:
        st.error("Unable to load statistics")
        st.info("Check API connection and try again")

    # Show detection service status
    st.markdown("---")
    st.markdown("### 🔧 Service Status")

    with st.spinner("Checking service status..."):
        status_response = api_client.get_detection_status()

        if status_response.get("success"):
            status_data = status_response
            st.success("🟢 Detection Service Online")

            # Show model information
            if status_data.get("model_loaded"):
                st.info("✅ Model Loaded")
            else:
                st.warning("⚠️ Model Not Loaded")

            # Show supported formats
            formats = status_response.get("supported_formats", [])
            if formats:
                st.markdown(f"**Supported formats:** {', '.join(formats)}")

            # Show file size limit
            max_size = status_response.get("max_file_size", 0)
            if max_size > 0:
                st.markdown(f"**Max file size:** {format_file_size(max_size)}")

            # Show confidence threshold
            threshold = status_data.get("confidence_threshold", 0.6)
            st.markdown(f"**Default confidence:** {threshold:.1f}")
        else:
            st.error("🔴 Detection Service Offline")
            st.warning("The disease detection service is currently unavailable")


def show_detection_history():
    """Display recent detection history"""
    st.markdown("---")
    st.markdown("### 📜 Detection History")

    # Get history from session state
    history = st.session_state.get('detection_history', [])

    if not history:
        show_empty_state(
            icon="📷",
            title="No Detection History",
            description="Your disease detection results will appear here once you start analyzing images.",
            action_text="Upload Your First Image",
            action_callback=lambda: st.session_state.update({"nav_select": "Home"})
        )
        return

    # Show recent detections
    recent_history = history[:5]  # Show last 5

    for i, detection in enumerate(reversed(recent_history)):
        with st.expander(
            f"📸 {detection.get('disease_name', 'Unknown')} "
            f"({detection.get('confidence', 0):.1% confidence}) - "
            f"{format_timestamp(detection.get('timestamp', ''))}"
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                # Show detection details
                st.markdown(f"**Disease:** {detection.get('disease_name', 'Unknown')}")
                st.markdown(f"**Scientific Name:** {detection.get('scientific_name', 'N/A')}")
                st.markdown(f"**Confidence:** {detection.get('confidence', 0):.1%}")

                if detection.get('severity'):
                    st.markdown(f"**Severity:** {create_severity_badge(detection.get('severity'))}")

                if detection.get('processing_time'):
                    st.markdown(f"**Processing Time:** {format_processing_time(detection.get('processing_time', 0))}")

            with col2:
                # Show confidence gauge
                if detection.get('confidence'):
                    create_confidence_gauge(detection.get('confidence', 0))

                # Show view details button
                if st.button(f"View Details", key=f"view_details_{i}"):
                    # Set the detection result for detailed view
                    st.session_state.selected_detection = detection
                    st.session_state.current_page = "Results"
                    st.rerun()

    # Show clear history button
    if history:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.detection_history = []
                st.session_state.selected_detection = None
                st.success("History cleared successfully!")
                time.sleep(1)
                st.rerun()


def perform_disease_detection(
    uploaded_file,
    confidence_threshold: float,
    enhance_contrast: bool,
    include_history: bool
):
    """Perform disease detection on uploaded image"""
    # Validate file
    api_client = st.session_state.api_client
    is_valid, error_msg = api_client.validate_image_file(uploaded_file)

    if not is_valid:
        show_error_message(error_msg)
        return

    # Show processing state
    processing_placeholder = st.empty()
    processing_placeholder.markdown("🔄 **Processing...** Analyzing image for disease detection")

    try:
        # Perform detection
        start_time = time.time()
        result = api_client.detect_disease(uploaded_file, confidence_threshold)
        processing_time = time.time() - start_time

        # Clear processing placeholder
        processing_placeholder.empty()

        if result.get("success"):
            # Success - store results
            formatted_result = api_client.format_detection_result(result)
            formatted_result['processing_time'] = processing_time

            # Store in session state
            st.session_state.detection_results = formatted_result
            st.session_state.uploaded_image = uploaded_file

            if include_history:
                # Save to history
                history_item = {
                    'disease_name': formatted_result.get('disease_name'),
                    'scientific_name': formatted_result.get('scientific_name'),
                    'confidence': formatted_result.get('confidence'),
                    'severity': formatted_result.get('severity'),
                    'timestamp': formatted_result.get('detected_at'),
                    'processing_time': processing_time,
                    'image_info': formatted_result.get('image_info'),
                    'class_index': formatted_result.get('class_index')
                }
                api_client.save_request_to_history(history_item)

            # Show success message
            show_success_message(f"Disease detected: {formatted_result.get('disease_name')}")

            # Navigate to results page after a short delay
            time.sleep(2)
            st.session_state.current_page = "Results"
            st.rerun()

        else:
            # Error in detection
            error_msg = api_client.format_error_message(result)
            show_error_message(error_msg)

            # Show additional error details if available
            if result.get("error") and "confident" in result.get("error", "").lower():
                st.warning("""
                **Low Confidence Detection**

                The detection confidence was below your threshold. Try:
                - Using a clearer, more focused image
                - Adjusting the confidence threshold slider
                - Ensuring good lighting and focus
                """)

    except Exception as e:
        processing_placeholder.empty()
        show_error_message(f"An unexpected error occurred: {str(e)}")


def show_detailed_results(detection_result: Dict[str, Any], uploaded_file):
    """Show detailed detection results"""
    st.markdown("### 🔍 Detection Results")

    # Create two-column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Image and prediction
        st.markdown("#### Uploaded Image")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Analyzed Image", use_column_width=True)

        # Prediction results
        st.markdown("#### Prediction")

        disease_name = detection_result.get('disease_name', 'Unknown')
        scientific_name = detection_result.get('scientific_name', 'N/A')
        confidence = detection_result.get('confidence', 0)
        severity = detection_result.get('severity', 'Unknown')

        # Create result card
        st.markdown(f"""
            <div style='background-color: white; border-radius: 8px; padding: 1.5rem; border: 1px solid #E0E0E0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='color: #2E7D32; margin-top: 0;'>{disease_name}</h3>
                <p style='color: #666; font-style: italic; margin-bottom: 1rem;'>{scientific_name}</p>

                <div style='display: flex; gap: 1rem; margin-bottom: 1rem;'>
                    <div style='flex: 1;'>
                        <strong>Confidence:</strong>
                        <div style='margin-top: 0.5rem;'>{create_confidence_bar(confidence)}</div>
                    </div>
                    <div style='flex: 1;'>
                        <strong>Severity:</strong>
                        <div style='margin-top: 0.5rem;'>{create_severity_badge(severity)}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Processing info
        st.markdown("#### Analysis Information")
        processing_time = detection_result.get('processing_time', 0)
        image_info = detection_result.get('image_info', {})

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Processing Time", f"{format_processing_time(processing_time)}")
            st.metric("Image Size", format_file_size(image_info.get('size', 0)))

        with col2:
            dimensions = image_info.get('dimensions', [])
            if dimensions and len(dimensions) >= 2:
                st.metric("Dimensions", f"{dimensions[0]} × {dimensions[1]}")
            st.metric("Format", image_info.get('format', 'Unknown'))

        # Action buttons
        st.markdown("#### Actions")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📋 View Full Disease Info", key="view_disease_details"):
                # Navigate to disease details
                st.session_state.selected_disease = disease_name
                st.session_state.current_page = "Disease Database"
                st.rerun()

        with col2:
            if st.button("🔄 Analyze Another Image", key="analyze_another"):
                # Reset for new analysis
                st.session_state.detection_results = None
                st.session_state.uploaded_image = None
                st.rerun()

    with col2:
        # Confidence gauge
        st.markdown("#### Confidence Score")
        create_confidence_gauge(confidence, "Detection Confidence")

        # Additional info
        if detection_result.get('class_index') is not None:
            st.markdown("#### Classification Details")
            st.metric("Class Index", detection_result.get('class_index'))

        if detection_result.get('detected_at'):
            detected_time = format_timestamp(detection_result.get('detected_at'))
            st.markdown(f"**Detection Time:** {detected_time}")


def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp