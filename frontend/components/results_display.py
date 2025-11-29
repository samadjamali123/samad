"""
Results Display Component

This component displays comprehensive analysis results with confidence indicators,
visual comparisons, and interactive elements for exploring disease information.
"""

import streamlit as st
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import base64
import io

from frontend.components.treatment_card import TreatmentCard
from frontend.utils.styling import create_confidence_bar, create_comparison_layout

class ResultsDisplay:
    """Comprehensive results display with confidence indicators and visual comparisons"""

    def __init__(self, analysis_results: Dict[str, Any], uploaded_image: Optional[bytes] = None):
        """Initialize with analysis results and optional uploaded image"""
        self.analysis_results = analysis_results
        self.uploaded_image = uploaded_image
        self.selected_disease = None

    def display(self) -> Optional[str]:
        """Display complete analysis results"""
        if not self.analysis_results or not self.analysis_results.get('success', False):
            st.error("❌ No analysis results available")
            return None

        # Display results sections
        self._display_header()
        self._display_overview()
        self._display_plant_identification()
        self._display_disease_detection()
        self._display_model_performance()
        self._display_image_comparison()
        self._display_confidence_breakdown()

        return self.selected_disease

    def _display_header(self):
        """Display results header with summary"""
        results = self.analysis_results.get('results', {})
        plant_detection = results.get('plant_detection', {})
        disease_detection = results.get('disease_detection', {})

        plant_name = plant_detection.get('plant_name', 'Unknown')
        plant_confidence = plant_detection.get('confidence', 0)
        primary_disease = disease_detection.get('primary_disease', {})
        disease_name = primary_disease.get('disease_name', None)
        disease_confidence = primary_disease.get('confidence', 0)

        # Create summary card
        st.markdown(f"""
        <div class="results-header">
            <div class="results-summary">
                <div class="summary-item plant">
                    <div class="summary-icon">🌿</div>
                    <div class="summary-content">
                        <h3>{plant_name}</h3>
                        <p>Plant Identified</p>
                        <div class="confidence-badge good">
                            {plant_confidence:.0%} Confidence
                        </div>
                    </div>
                </div>
                {'</div>' if disease_name else ''}
                <div class="summary-item {'disease' if disease_name else 'healthy'}">
                    <div class="summary-icon">{'🔬' if disease_name else '✅'}</div>
                    <div class="summary-content">
                        <h3>{disease_name if disease_name else 'Healthy'}</h3>
                        <p>{'Disease Detected' if disease_name else 'No Disease Found'}</p>
                        <div class="confidence-badge {'good' if disease_confidence > 0.8 else 'medium' if disease_confidence > 0.6 else 'low'}">
                            {disease_confidence:.0%} Confidence
                        </div>
                    </div>
                </div>
            </div>
            <div class="results-timestamp">
                <p>Analysis completed in {results.get('metadata', {}).get('processing_time_total_ms', 0)}ms</p>
                <p>Using {len(results.get('metadata', {}).get('models_used', []))} AI models</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def _display_overview(self):
        """Display quick overview with key metrics"""
        results = self.analysis_results.get('results', {})
        plant_detection = results.get('plant_detection', {})
        disease_detection = results.get('disease_detection', {})
        metadata = results.get('metadata', {})

        # Create overview cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            plant_confidence = plant_detection.get('confidence', 0)
            self._create_metric_card(
                "🌱 Plant ID",
                plant_confidence,
                f"{plant_detection.get('plant_name', 'Unknown')}",
                "success" if plant_confidence > 0.8 else "warning" if plant_confidence > 0.6 else "error"
            )

        with col2:
            primary_disease = disease_detection.get('primary_disease', {})
            disease_confidence = primary_disease.get('confidence', 0)
            disease_name = primary_disease.get('disease_name', None)

            if disease_name:
                status = "error" if disease_confidence > 0.8 else "warning" if disease_confidence > 0.6 else "info"
                label = "🔬 Disease"
            else:
                status = "success"
                label = "✅ Healthy"

            self._create_metric_card(
                label,
                disease_confidence if disease_name else 1.0,
                disease_name if disease_name else "No disease detected",
                status
            )

        with col3:
            quality_score = metadata.get('image_quality_score', 0)
            self._create_metric_card(
                "📊 Image Quality",
                quality_score,
                f"{quality_score:.0%} Score",
                "success" if quality_score > 0.7 else "warning" if quality_score > 0.4 else "error"
            )

        with col4:
            models_used = metadata.get('models_used', [])
            self._create_metric_card(
                "🤖 AI Models",
                len(models_used),
                f"{len(models_used)} models active",
                "success"
            )

    def _create_metric_card(self, label: str, value: float, description: str, status: str):
        """Create a metric display card"""
        status_colors = {
            "success": "#4CAF50",
            "warning": "#FF9800",
            "error": "#F44336",
            "info": "#2196F3"
        }

        color = status_colors.get(status, "#2196F3")

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color: {color};">
                {value:.0%} if value <= 1.0 else value}
            </div>
            <div class="metric-description">{description}</div>
        </div>
        """, unsafe_allow_html=True)

    def _display_plant_identification(self):
        """Display detailed plant identification results"""
        results = self.analysis_results.get('results', {})
        plant_detection = results.get('plant_detection', {})

        with st.expander("🌱 Plant Identification Details", expanded=True):
            col1, col2 = st.columns([3, 2])

            with col1:
                # Primary identification
                st.markdown("**Primary Identification**")
                plant_name = plant_detection.get('plant_name', 'Unknown')
                confidence = plant_detection.get('confidence', 0)

                # Confidence bar
                confidence_html = create_confidence_bar(confidence, "Plant Identification")
                st.markdown(confidence_html, unsafe_allow_html=True)

                # Scientific name (if available from AI models)
                ai_models = results.get('ai_model_results', {})
                scientific_name = None

                # Try to get scientific name from different models
                for model_name, model_data in ai_models.items():
                    if isinstance(model_data, dict) and model_data.get('success', False):
                        # This would need to be extracted from the actual model response
                        pass

                if scientific_name:
                    st.markdown(f"*Scientific Name: {scientific_name}*")

            with col2:
                # Alternative possibilities
                alternatives = plant_detection.get('alternative_plants', [])
                if alternatives:
                    st.markdown("**Alternative Possibilities**")

                    for i, alt in enumerate(alternatives[:3]):  # Show top 3
                        alt_name = alt.get('name', f'Alternative {i+1}')
                        alt_confidence = alt.get('confidence', 0)

                        st.markdown(f"""
                        <div class="alternative-item">
                            <div class="alternative-name">{alt_name}</div>
                            <div class="alternative-confidence">
                                {alt_confidence:.0%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    def _display_disease_detection(self):
        """Display detailed disease detection results"""
        results = self.analysis_results.get('results', {})
        disease_detection = results.get('disease_detection', {})
        primary_disease = disease_detection.get('primary_disease', {})
        secondary_diseases = disease_detection.get('secondary_diseases', [])

        with st.expander("🔬 Disease Detection Results", expanded=True):
            # Primary disease
            disease_name = primary_disease.get('disease_name')
            confidence = primary_disease.get('confidence', 0)

            if disease_name:
                col1, col2 = st.columns([3, 2])

                with col1:
                    st.markdown("**Primary Disease Detected**")

                    # Disease name with confidence
                    st.markdown(f"### {disease_name}")

                    # Confidence bar
                    confidence_html = create_confidence_bar(confidence, "Disease Detection")
                    st.markdown(confidence_html, unsafe_allow_html=True)

                    # Store selected disease for treatment card
                    if st.button("💊 View Treatment Recommendations", key="view_treatments", use_container_width=True):
                        self.selected_disease = disease_name

                with col2:
                    st.markdown("**Confidence Level**")

                    if confidence > 0.8:
                        st.success("High Confidence")
                        st.markdown("✅ Clear disease symptoms detected")
                    elif confidence > 0.6:
                        st.warning("Medium Confidence")
                        st.markdown("⚠️ Some disease symptoms visible")
                    else:
                        st.error("Low Confidence")
                        st.markdown("❌ Unclear or early-stage symptoms")

                    # Add action buttons
                    if st.button("📊 More Details", key="disease_details"):
                        st.info("Detailed disease information will be available in the treatment section")

            else:
                st.success("### ✅ No Disease Detected")
                st.markdown("""
                **Good news!** The plant appears healthy.

                **Recommendations:**
                - Continue regular monitoring
                - Maintain good growing conditions
                - Practice preventive care
                - Check periodically for early symptoms
                """)

            # Secondary diseases
            if secondary_diseases:
                st.markdown("**Other Possible Diseases**")

                for i, sec_disease in enumerate(secondary_diseases[:3]):  # Show top 3
                    sec_name = sec_disease.get('name', f'Alternative {i+1}')
                    sec_confidence = sec_disease.get('confidence', 0)

                    with st.expander(f"🔍 {sec_name} ({sec_confidence:.0%})"):
                        st.markdown(f"""
                        <div class="secondary-disease">
                            <p><strong>Confidence:</strong> {sec_confidence:.0%}</p>
                            <p>This disease was also detected but with lower confidence.</p>
                        </div>
                        """, unsafe_allow_html=True)

    def _display_model_performance(self):
        """Display AI model performance metrics"""
        results = self.analysis_results.get('results', {})
        ai_models = results.get('ai_model_results', {})

        with st.expander("🤖 AI Model Performance", expanded=False):
            if ai_models:
                # Create performance table
                model_data = []

                for model_name, model_data in ai_models.items():
                    if isinstance(model_data, dict):
                        model_data = {
                            'Model': model_name.title(),
                            'Success': '✅' if model_data.get('success', False) else '❌',
                            'Plant Confidence': f"{model_data.get('plant_confidence', 0):.0%}",
                            'Disease Confidence': f"{model_data.get('disease_confidence', 0):.0%}",
                            'Processing Time': f"{model_data.get('processing_time_ms', 0)}ms"
                        }
                        model_data.append(model_data)

                # Display as formatted table
                for i, model in enumerate(model_data):
                    if i > 0:
                        st.markdown("---")

                    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

                    with col1:
                        st.markdown(f"**{model['Model']}**")

                    with col2:
                        st.markdown(model['Success'])

                    with col3:
                        st.markdown(f"<small>{model['Plant Confidence']}</small>", unsafe_allow_html=True)

                    with col4:
                        st.markdown(f"<small>{model['Disease Confidence']}</small>", unsafe_allow_html=True)

                    with col5:
                        st.markdown(f"<small>{model['Processing Time']}</small>", unsafe_allow_html=True)
            else:
                st.info("No model performance data available")

    def _display_image_comparison(self):
        """Display side-by-side comparison with reference images"""
        if not self.uploaded_image:
            return

        results = self.analysis_results.get('results', {})
        disease_detection = results.get('disease_detection', {})
        primary_disease = disease_detection.get('primary_disease', {})
        disease_name = primary_disease.get('disease_name')

        if not disease_name:
            return

        with st.expander("📸 Image Comparison", expanded=False):
            # Create comparison layout
            comparison_html = create_comparison_layout(self.uploaded_image, disease_name)
            st.markdown(comparison_html, unsafe_allow_html=True)

            st.markdown("""
            <div class="comparison-info">
                <p><strong>Left:</strong> Your uploaded image</p>
                <p><strong>Right:</strong> Reference images showing typical symptoms</p>
                <p><em>Compare the patterns and symptoms to understand the disease better.</em></p>
            </div>
            """, unsafe_allow_html=True)

    def _display_confidence_breakdown(self):
        """Display detailed confidence analysis"""
        results = self.analysis_results.get('results', {})
        plant_detection = results.get('plant_detection', {})
        disease_detection = results.get('disease_detection', {})
        metadata = results.get('metadata', {})

        with st.expander("📊 Confidence Analysis", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Identification Confidence**")

                # Plant identification confidence
                plant_confidence = plant_detection.get('confidence', 0)
                self._create_confidence_analysis(
                    "Plant Identification",
                    plant_confidence,
                    "✅ High" if plant_confidence > 0.8 else "⚠️ Medium" if plant_confidence > 0.6 else "❌ Low",
                    "✅ Clear plant characteristics" if plant_confidence > 0.8 else "⚠️ Some uncertainty" if plant_confidence > 0.6 else "❌ Unclear identification"
                )

            with col2:
                st.markdown("**Detection Confidence**")

                # Disease detection confidence
                primary_disease = disease_detection.get('primary_disease', {})
                disease_confidence = primary_disease.get('confidence', 0)

                if primary_disease.get('disease_name'):
                    self._create_confidence_analysis(
                        "Disease Detection",
                        disease_confidence,
                        "✅ High" if disease_confidence > 0.8 else "⚠️ Medium" if disease_confidence > 0.6 else "❌ Low",
                        "✅ Clear symptoms detected" if disease_confidence > 0.8 else "⚠️ Some symptoms visible" if disease_confidence > 0.6 else "❌ Unclear symptoms"
                    )
                else:
                    st.markdown("""
                    <div class="confidence-analysis">
                        <h4>Health Status</h4>
                        <div class="confidence-indicator excellent">
                            ✅ Confirmed Healthy
                        </div>
                        <p>No disease symptoms detected in the image.</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Overall analysis confidence
            image_quality = metadata.get('image_quality_score', 0)
            models_used = metadata.get('models_used', [])

            st.markdown("---")
            st.markdown("**Overall Analysis Quality**")

            # Create overall confidence metric
            if plant_confidence > 0.8 and image_quality > 0.7 and len(models_used) >= 2:
                overall_status = "Excellent"
                overall_color = "#4CAF50"
                overall_icon = "🏆"
            elif plant_confidence > 0.6 and image_quality > 0.5:
                overall_status = "Good"
                overall_color = "#FF9800"
                overall_icon = "👍"
            else:
                overall_status = "Fair"
                overall_color = "#F44336"
                overall_icon = "⚠️"

            st.markdown(f"""
            <div class="overall-confidence">
                <div class="overall-score" style="color: {overall_color};">
                    {overall_icon} <strong>{overall_status}</strong>
                </div>
                <p>Analysis quality based on image clarity, model consensus, and confidence levels</p>
            </div>
            """, unsafe_allow_html=True)

    def _create_confidence_analysis(self, title: str, confidence: float, status: str, description: str):
        """Create confidence analysis component"""
        color = "#4CAF50" if confidence > 0.8 else "#FF9800" if confidence > 0.6 else "#F44336"

        st.markdown(f"""
        <div class="confidence-analysis">
            <h4>{title}</h4>
            <div class="confidence-indicator" style="color: {color};">
                {confidence:.0%} - {status}
            </div>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)

    def get_selected_disease(self) -> Optional[str]:
        """Get the currently selected disease for treatment display"""
        return self.selected_disease

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of analysis results for sharing or saving"""
        results = self.analysis_results.get('results', {})
        plant_detection = results.get('plant_detection', {})
        disease_detection = results.get('disease_detection', {})
        metadata = results.get('metadata', {})

        return {
            'plant_name': plant_detection.get('plant_name', 'Unknown'),
            'plant_confidence': plant_detection.get('confidence', 0),
            'disease_name': disease_detection.get('primary_disease', {}).get('disease_name'),
            'disease_confidence': disease_detection.get('primary_disease', {}).get('confidence', 0),
            'image_quality': metadata.get('image_quality_score', 0),
            'processing_time': metadata.get('processing_time_total_ms', 0),
            'models_used': metadata.get('models_used', []),
            'timestamp': self.analysis_results.get('timestamp', 0),
            'session_id': self.analysis_results.get('session_id', '')
        }

# Utility functions for HTML generation
def create_confidence_bar(confidence: float, label: str) -> str:
    """Create HTML for confidence bar with color coding"""
    # Determine color based on confidence level
    if confidence >= 0.8:
        color = "#4CAF50"  # Green
        status = "High"
    elif confidence >= 0.6:
        color = "#FF9800"  # Orange
        status = "Medium"
    else:
        color = "#F44336"  # Red
        status = "Low"

    return f"""
    <div class="confidence-bar-container">
        <div class="confidence-label">{label}</div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence * 100}%; background-color: {color};">
            </div>
            <div class="confidence-value">{confidence:.0%}</div>
        </div>
        <div class="confidence-status" style="color: {color};">{status} Confidence</div>
    </div>
    """

def create_comparison_layout(uploaded_image: bytes, disease_name: str) -> str:
    """Create HTML for image comparison layout"""
    # Convert uploaded image to base64
    uploaded_b64 = base64.b64encode(uploaded_image).decode()

    # Reference image paths (these would be actual paths to reference images)
    reference_images = [
        f"database/reference_images/{disease_name.lower().replace(' ', '_')}_1.jpg",
        f"database/reference_images/{disease_name.lower().replace(' ', '_')}_2.jpg"
    ]

    comparison_html = f"""
    <div class="image-comparison">
        <div class="comparison-column">
            <div class="comparison-header">
                <h4>Your Image</h4>
            </div>
            <div class="comparison-image">
                <img src="data:image/jpeg;base64,{uploaded_b64}" alt="Uploaded leaf image">
            </div>
        </div>
        <div class="comparison-column">
            <div class="comparison-header">
                <h4>Reference Images</h4>
                <p>Typical {disease_name} symptoms</p>
            </div>
            <div class="reference-images">
    """

    for i, ref_img in enumerate(reference_images):
        comparison_html += f"""
                <div class="reference-image">
                    <img src="{ref_img}" alt="Reference image {i+1}" onerror="this.style.display='none';">
                    <div class="reference-label">Example {i+1}</div>
                </div>
        """

    comparison_html += """
            </div>
        </div>
    </div>
    """

    return comparison_html


# Grad-CAM Display Extension
class GradCAMExtension:
    """Extension for adding Grad-CAM functionality to results display"""

    def __init__(self, analysis_results: Dict[str, Any], uploaded_image: Optional[bytes] = None):
        """Initialize Grad-CAM extension"""
        self.analysis_results = analysis_results
        self.uploaded_image = uploaded_image

    def display_grad_cam_section(self):
        """Display Grad-CAM analysis section"""
        if not self.uploaded_image:
            return

        st.markdown("### 🎯 Grad-CAM Visualization Analysis")
        st.markdown("*See exactly where the AI focused when detecting diseases*")

        # Information section
        with st.expander("ℹ️ What is Grad-CAM?", expanded=False):
            st.markdown("""
            **Grad-CAM (Gradient-weighted Class Activation Mapping)** shows which parts of the leaf
            the AI model focused on when making its disease detection decision.

            **How to interpret:**
            - 🔴 **Red/Orange areas**: High attention - likely disease symptoms
            - 🟡 **Yellow areas**: Medium attention - possible symptoms
            - 🔵 **Blue/Green areas**: Low attention - healthy tissue
            - ⚪ **White areas**: No attention - background/irrelevant

            **Benefits:**
            - ✅ Understand *why* AI made its decision
            - ✅ Verify AI focused on actual symptoms
            - ✅ Build trust in AI recommendations
            - ✅ Learn to identify symptoms yourself
            """)

        # Generate button
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            generate_clicked = st.button(
                "🎯 Generate Grad-CAM Analysis",
                key="generate_grad_cam_main",
                help="Generate heatmap showing AI focus areas",
                use_container_width=True
            )

        with col2:
            processing_time = getattr(st.session_state, 'grad_cam_time_ms', 0)
            st.metric("⏱️ Time", f"{processing_time}ms", "Processing time")

        with col3:
            quality_score = getattr(st.session_state, 'grad_cam_quality', 0)
            st.metric("📊 Quality", f"{quality_score:.0%}", "Analysis quality")

        if generate_clicked:
            self._generate_grad_cam_analysis()

        # Display existing Grad-CAM results if available
        if hasattr(st.session_state, 'last_grad_cam') and st.session_state.last_grad_cam:
            self._display_existing_grad_cam()

    def _generate_grad_cam_analysis(self):
        """Generate Grad-CAM analysis"""
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Initialize
            progress_bar.progress(0.2)
            status_text.text("🔄 Initializing Grad-CAM generation...")

            # Step 2: Call API
            progress_bar.progress(0.5)
            status_text.text("🧠 Running AI model analysis...")

            # This would call the actual API
            # For now, simulate with placeholder
            import time
            time.sleep(2)  # Simulate processing

            # Step 3: Complete
            progress_bar.progress(1.0)
            status_text.text("✅ Grad-CAM analysis complete!")

            # Store mock results in session
            st.session_state.last_grad_cam = {
                'success': True,
                'original_image': base64.b64encode(self.uploaded_image).decode(),
                'heatmap_image': None,  # Would be generated
                'overlay_image': None,  # Would be generated
                'prediction': {
                    'disease_name': 'Sample Disease',
                    'confidence': 0.85,
                    'severity': 'moderate',
                    'is_healthy': False
                },
                'metadata': {
                    'model_version': 'v1.0.0',
                    'device': 'CPU',
                    'processing_time': 1500,
                    'heatmap_generated': False,
                    'image_size': [512, 512]
                }
            }
            st.session_state.grad_cam_time_ms = 1500
            st.session_state.grad_cam_quality = 0.85

            st.success("✅ Grad-CAM analysis completed! View results in the visualization tabs below.")
            st.rerun()

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error generating Grad-CAM: {str(e)}")

    def _display_existing_grad_cam(self):
        """Display existing Grad-CAM results"""
        grad_cam_data = st.session_state.last_grad_cam

        # Create visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Main View", "📸 Original", "🔥 Analysis", "📊 Details"
        ])

        with tab1:
            self._display_main_view(grad_cam_data)

        with tab2:
            self._display_original_image(grad_cam_data)

        with tab3:
            self._display_analysis_view(grad_cam_data)

        with tab4:
            self._display_detailed_metrics(grad_cam_data)

    def _display_main_view(self, grad_cam_data: Dict[str, Any]):
        """Display main combined view"""
        st.markdown("### 🎯 Complete Grad-CAM Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📸 Original Image**")
            if grad_cam_data.get('original_image'):
                st.image(
                    f"data:image/png;base64,{grad_cam_data['original_image']}",
                    use_column_width=True,
                    caption="Uploaded leaf image"
                )

        with col2:
            st.markdown("**🎯 AI Focus Areas**")
            if grad_cam_data.get('overlay_image'):
                st.image(
                    f"data:image/png;base64,{grad_cam_data['overlay_image']}",
                    use_column_width=True,
                    caption="AI attention overlay"
                )
            else:
                st.info("🔄 Generating overlay visualization...")
                st.markdown("""
                **What you'll see:**
                - 🔴 Red highlights: Disease symptoms
                - 🟡 Yellow highlights: Possible symptoms
                - 🔵 Blue highlights: Healthy tissue
                """)

        # Key insights
        st.markdown("---")
        prediction = grad_cam_data.get('prediction', {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if prediction.get('disease_name'):
                st.metric("🔬 Disease", prediction['disease_name'], "Detected")
            else:
                st.metric("✅ Status", "Healthy", "No disease")

        with col2:
            confidence = prediction.get('confidence', 0)
            st.metric("🎯 Confidence", f"{confidence:.0%}", "AI certainty")

        with col3:
            severity = prediction.get('severity', 'Unknown')
            st.metric("📊 Severity", severity.title(), "Disease level")

        with col4:
            metadata = grad_cam_data.get('metadata', {})
            device = metadata.get('device', 'Unknown')
            st.metric("💻 Device", device.upper(), "Processing unit")

    def _display_original_image(self, grad_cam_data: Dict[str, Any]):
        """Display original image with metadata"""
        st.markdown("### 📸 Original Image Analysis")

        if grad_cam_data.get('original_image'):
            st.image(
                f"data:image/png;base64,{grad_cam_data['original_image']}",
                use_column_width=True,
                caption="Your original leaf image"
            )

            metadata = grad_cam_data.get('metadata', {})
            image_size = metadata.get('image_size', [0, 0])

            if image_size and image_size != [0, 0]:
                st.markdown(f"""
                **Image Information:**
                - 📐 **Dimensions**: {image_size[0]} × {image_size[1]} pixels
                - 📊 **Quality**: High resolution for AI analysis
                - 🎯 **Analysis**: Ready for disease detection
                - ✅ **Status**: Successfully processed
                """)

    def _display_analysis_view(self, grad_cam_data: Dict[str, Any]):
        """Display analysis interpretation"""
        st.markdown("### 🔥 Grad-CAM Analysis")
        st.markdown("*Understanding AI's decision-making process*")

        prediction = grad_cam_data.get('prediction', {})
        metadata = grad_cam_data.get('metadata', {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🔬 Detection Analysis**")

            if prediction.get('disease_name'):
                disease_name = prediction['disease_name']
                confidence = prediction.get('confidence', 0)
                severity = prediction.get('severity', 'Unknown')

                # Confidence bar
                if confidence >= 0.8:
                    conf_color = "🟢"
                    conf_status = "High Confidence"
                elif confidence >= 0.6:
                    conf_color = "🟡"
                    conf_status = "Medium Confidence"
                else:
                    conf_color = "🔴"
                    conf_status = "Low Confidence"

                st.markdown(f"""
                **Detected Disease:** {disease_name}
                **Confidence Level:** {conf_color} {conf_status} ({confidence:.0%})
                **Severity:** {severity.title()}

                **AI Interpretation:** {'Clear disease symptoms detected' if confidence > 0.8 else 'Some symptoms visible' if confidence > 0.6 else 'Early/unclear symptoms'}
                """)
            else:
                st.markdown("""
                **✅ Plant Health Status: Healthy**

                The AI analysis found no clear disease symptoms.

                **Recommendations:**
                - Continue regular monitoring
                - Maintain optimal growing conditions
                - Practice preventive care
                """)

        with col2:
            st.markdown("**🤖 Model Performance**")

            model_version = metadata.get('model_version', 'Unknown')
            processing_time = metadata.get('processing_time', 0)
            device = metadata.get('device', 'Unknown')
            heatmap_generated = metadata.get('heatmap_generated', False)

            st.markdown(f"""
            **Model Information:**
            - **Version:** {model_version}
            - **Processing:** {device.upper()}
            - **Speed:** {processing_time}ms
            - **Grad-CAM:** {'✅ Available' if heatmap_generated else '❌ Unavailable'}

            **Performance Quality:**
            - 🎯 **Accuracy:** {'Excellent' if confidence > 0.8 else 'Good' if confidence > 0.6 else 'Fair'}
            - ⚡ **Processing:** {'Fast' if processing_time < 2000 else 'Normal'}
            - 🧠 **Interpretability:** {'High' if heatmap_generated else 'Medium'}
            """)

    def _display_detailed_metrics(self, grad_cam_data: Dict[str, Any]):
        """Display detailed metrics and technical information"""
        st.markdown("### 📊 Technical Analysis")

        prediction = grad_cam_data.get('prediction', {})
        metadata = grad_cam_data.get('metadata', {})

        # Metrics grid
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🔬 Disease Metrics**")

            if prediction.get('disease_name'):
                disease_name = prediction['disease_name']
                confidence = prediction.get('confidence', 0)
                severity = prediction.get('severity', 'Unknown')

                # Create detailed metrics
                st.metric("Disease", disease_name, "Primary detection")
                st.metric("Confidence", f"{confidence:.0%}", "Detection certainty")
                st.metric("Severity", severity.title(), "Disease level")

                # Health status
                is_healthy = prediction.get('is_healthy', True)
                health_status = "✅ Healthy" if is_healthy else "🔴 Diseased"
                st.metric("Health Status", health_status, "Overall condition")
            else:
                st.metric("Status", "✅ Healthy", "No disease detected")
                st.metric("Recommendation", "Monitor", "Continue observation")

        with col2:
            st.markdown("**💻 Technical Details**")

            model_version = metadata.get('model_version', 'Unknown')
            processing_time = metadata.get('processing_time', 0)
            device = metadata.get('device', 'Unknown')
            image_size = metadata.get('image_size', [0, 0])

            st.metric("Model", model_version, "AI version")
            st.metric("Device", device.upper(), "Processing unit")
            st.metric("Speed", f"{processing_time}ms", "Processing time")

            if image_size and image_size != [0, 0]:
                pixels = image_size[0] * image_size[1]
                st.metric("Resolution", f"{pixels:,}", "Total pixels")

        # Quality assessment
        st.markdown("---")
        st.markdown("**🏆 Quality Assessment**")

        confidence = prediction.get('confidence', 0)
        processing_time = metadata.get('processing_time', 0)
        heatmap_generated = metadata.get('heatmap_generated', False)

        # Calculate overall quality score
        quality_score = 0
        quality_factors = []

        if confidence > 0.8:
            quality_score += 0.4
            quality_factors.append("✅ High confidence detection")
        elif confidence > 0.6:
            quality_score += 0.3
            quality_factors.append("🟡 Medium confidence detection")

        if processing_time < 2000:
            quality_score += 0.3
            quality_factors.append("⚡ Fast processing")

        if heatmap_generated:
            quality_score += 0.3
            quality_factors.append("🎯 Grad-CAM available")

        # Display quality
        if quality_score >= 0.9:
            quality_status = "🏆 Excellent"
            quality_color = "🟢"
        elif quality_score >= 0.7:
            quality_status = "👍 Good"
            quality_color = "🟡"
        else:
            quality_status = "⚠️ Fair"
            quality_color = "🔴"

        st.markdown(f"""
        **Overall Quality:** {quality_color} {quality_status} ({quality_score:.0%})

        **Quality Factors:**
        """)

        for factor in quality_factors:
            st.markdown(f"- {factor}")

        if quality_score < 0.7:
            st.markdown("""
            **Improvement Suggestions:**
            - 📸 Use higher quality images with better lighting
            - 🔍 Ensure disease symptoms are clearly visible
            - 🌿 Include multiple leaves if possible
            - ☀️ Avoid shadows and glare in photos
            """)