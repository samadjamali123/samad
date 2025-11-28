"""
Results Page - Detailed Disease Detection Results
"""

import streamlit as st
from PIL import Image
from datetime import datetime
from utils.api_client import APIClient
from utils.ui_helpers import (
    show_page_header,
    show_empty_state,
    create_confidence_gauge,
    create_disease_card,
    create_severity_badge,
    format_file_size,
    format_processing_time,
    format_timestamp,
    show_error_message,
    show_success_message
)


def show_results_page():
    """Display detailed detection results"""
    show_page_header("📊 Detection Results", "Detailed analysis of your uploaded plant leaf")

    # Check if we have results
    if 'detection_results' not in st.session_state or not st.session_state.detection_results:
        show_empty_state(
            icon="📷",
            title="No Detection Results",
            description="Please upload a plant leaf image from the Home page to see detection results here.",
            action_text="Go to Upload Page",
            action_callback=lambda: st.session_state.update({"current_page": "Home"})
        )
        return

    # Get results
    detection_result = st.session_state.detection_results
    uploaded_image = st.session_state.get('uploaded_image')

    # Show detailed results
    if detection_result.get('success'):
        show_detailed_results(detection_result, uploaded_image)
    else:
        show_error_results(detection_result)


def show_detailed_results(detection_result, uploaded_image):
    """Display successful detection results"""
    # Main results layout
    col1, col2 = st.columns([3, 2])

    with col1:
        # Prediction results
        st.markdown("### 🔍 Disease Detection Results")

        # Create result card
        disease_name = detection_result.get('disease_name', 'Unknown')
        scientific_name = detection_result.get('scientific_name', '')
        confidence = detection_result.get('confidence', 0)
        severity = detection_result.get('severity', 'Unknown')
        detected_at = detection_result.get('detected_at', '')

        # Main disease card
        st.markdown(f"""
            <div style='background-color: white; border-radius: 8px; padding: 1.5rem; border: 1px solid #E0E0E0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
                <div style='background-color: #2E7D32; color: white; padding: 0.75rem 1rem; border-radius: 8px 8px 0 0; margin: -1.5rem -1.5rem 1rem -1.5rem;'>
                    <h3 style='margin: 0; color: white; font-size: 1.5rem;'>{disease_name}</h3>
                    <div style='margin-top: 0.5rem;'>{create_severity_badge(severity)}</div>
                </div>
                <p style='color: #666; font-style: italic; margin: 1rem 0;'>{scientific_name}</p>
                <div style='display: flex; gap: 1rem; margin: 1rem 0;'>
                    <div style='flex: 1;'>
                        <strong style='color: #2E7D32;'>Confidence:</strong>
                        <div style='margin-top: 0.5rem;'>{confidence:.1%}</div>
                    </div>
                    <div style='flex: 1;'>
                        <strong style='color: #2E7D32;'>Detected:</strong>
                        <div style='margin-top: 0.5rem; color: #666;'>{format_timestamp(detected_at)}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Processing information
        st.markdown("### 📈 Analysis Information")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Processing Time", f"{detection_result.get('processing_time', 0):.2f} seconds")
            if detection_result.get('class_index') is not None:
                st.metric("Class Index", str(detection_result.get('class_index')))

        with col2:
            image_info = detection_result.get('image_info', {})
            st.metric("Image Size", format_file_size(image_info.get('size', 0)))
            dimensions = image_info.get('dimensions', [])
            if dimensions and len(dimensions) >= 2:
                st.metric("Dimensions", f"{dimensions[0]} × {dimensions[1]}")

    with col2:
        # Confidence gauge
        st.markdown("### 📊 Confidence Score")
        create_confidence_gauge(confidence, "Detection Confidence")

        # Action buttons
        st.markdown("### 🔧 Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌿 View Disease Database", use_container_width=True, key="view_disease_db"):
                st.session_state.current_page = "Disease Database"
                st.rerun()

        with col2:
            if st.button("🔄 Analyze New Image", use_container_width=True, key="analyze_new"):
                st.session_state.detection_results = None
                st.session_state.uploaded_image = None
                st.session_state.current_page = "Home"
                st.rerun()

    # Show uploaded image
    if uploaded_image:
        st.markdown("---")
        st.markdown("### 📷 Analyzed Image")
        try:
            image = Image.open(uploaded_image)
            st.image(image, caption=f"**{uploaded_image.name}**", use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")

    # Similar diseases section
    st.markdown("---")
    st.markdown("### 🔍 Related Diseases")

    # Get similar diseases from API
    if 'api_client' in st.session_state:
        api_client = st.session_state.api_client

        with st.spinner("Finding similar diseases..."):
            class_index = detection_result.get('class_index')
            if class_index is not None:
                # Search for related diseases
                search_response = api_client.search_diseases(disease_name, limit=5)

                if search_response.get('success') and search_response.get('diseases'):
                    related_diseases = search_response.get('diseases', [])

                    for disease in related_diseases[:4]:  # Show top 4 related
                        with st.expander(f"🌱 {disease.get('name', 'Unknown')} ({disease.get('severity', 'Unknown')})"):
                            st.markdown(f"**Scientific Name:** {disease.get('scientific_name', 'N/A')}")
                            st.markdown(f"**Category:** {disease.get('category', 'Unknown')}")
                            st.markdown(f"**Severity:** {create_severity_badge(disease.get('severity', 'Unknown'))}")

                            affected_plants = disease.get('affected_plants', [])
                            if affected_plants:
                                st.markdown(f"**Affected Plants:** {', '.join(affected_plants[:3])}")
                                if len(affected_plants) > 3:
                                    st.markdown(f"... and {len(affected_plants) - 3} more")
                else:
                    st.info("No related diseases found")
            else:
                st.info("Unable to find related diseases at the moment")


def show_error_results(detection_result):
    """Display error detection results"""
    st.markdown("### ❌ Detection Failed")

    error_message = detection_result.get('error', 'Unknown error occurred')
    show_error_message(error_message)

    # Troubleshooting tips
    st.markdown("### 🔧 Troubleshooting Tips")

    tips = [
        "**Try a different image** - Ensure the leaf is clearly visible and in focus",
        "**Check lighting** - Natural daylight works best for disease detection",
        "**Adjust confidence threshold** - Lower the threshold if no diseases are detected",
        "**Image quality** - Use high-resolution images without blur or distortion",
        "**Multiple angles** - Try different views of the same leaf if possible"
    ]

    for tip in tips:
        st.markdown(tip)

    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Try Again", use_container_width=True, type="primary", key="try_again"):
            st.session_state.detection_results = None
            st.session_state.uploaded_image = None
            st.session_state.current_page = "Home"
            st.rerun()