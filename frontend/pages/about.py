"""
About Page - System Information and Documentation
"""

import streamlit as st
import os
from datetime import datetime
from utils.ui_helpers import (
    show_page_header,
    create_metric_card
)


def show_about_page():
    """Display system information and documentation"""
    show_page_header("ℹ️ About", "Plant Disease Detection System Information")

    # System Overview
    show_system_overview()

    # Features
    show_features()

    # Technology Stack
    show_technology_stack()

    # How It Works
    show_how_it_works()

    # Supported Diseases
    show_supported_diseases()

    # API Information
    show_api_information()

    # System Requirements
    show_system_requirements()

    # Contact and Support
    show_contact_information()


def show_system_overview():
    """Show system overview section"""
    st.markdown("### 🌿 System Overview")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
            **Plant Disease Detection System** is an AI-powered web application that helps farmers, gardeners, and plant enthusiasts identify plant diseases from leaf images.

            The system uses advanced machine learning models to analyze plant leaf images and provide:
            - **Disease identification** with confidence scores
            - **Comprehensive disease information** including causes, symptoms, and treatments
            - **Treatment recommendations** for both chemical and organic approaches
            - **Prevention strategies** to avoid future infections
            - **Database of plant diseases** for educational purposes
        """)

    with col2:
        # System metrics
        create_metric_card(
            "Version",
            "1.0.0"
        )

        create_metric_card(
            "Status",
            "🟢 Active"
        )

        create_metric_card(
            "API",
            "FastAPI"
        )

        create_metric_card(
            "Frontend",
            "Streamlit"
        )


def show_features():
    """Show system features"""
    st.markdown("---")
    st.markdown("### ✨ Key Features")

    features = [
        {
            "icon": "🔍",
            "title": "AI-Powered Detection",
            "description": "Advanced machine learning models for accurate disease identification with confidence scoring"
        },
        {
            "icon": "📊",
            "title": "Confidence Analysis",
            "description": "Real-time confidence scores and probability distributions for detection results"
        },
        {
            "icon": "🌱",
            "title": "Comprehensive Database",
            "description": "Extensive database of plant diseases with detailed information about symptoms, causes, and treatments"
        },
        {
            "icon": "🏥",
            "title": "Multiple Disease Categories",
            "description": "Support for fungal, bacterial, viral, nutritional, and environmental plant problems"
        },
        {
            "icon": "📱",
            "title": "Responsive Design",
            "description": "Mobile-friendly interface that works on desktop, tablet, and mobile devices"
        },
        {
            "icon": "🔄",
            "title": "Batch Processing",
            "description": "Analyze multiple images simultaneously for efficient disease screening"
        }
    ]

    # Feature cards in 2x3 grid
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
                <div style='background-color: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #E0E0E0; text-align: center; height: 100%;'>
                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>{feature['icon']}</div>
                    <h4 style='color: #2E7D32; margin: 0.5rem 0;'>{feature['title']}</h4>
                    <p style='color: #666; font-size: 0.9rem; margin: 0;'>{feature['description']}</p>
                </div>
            """, unsafe_allow_html=True)


def show_technology_stack():
    """Show technology stack information"""
    st.markdown("---")
    st.markdown("### 🛠️ Technology Stack")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Backend")
        st.markdown("""
            - **FastAPI** - Modern, fast web framework for building APIs
            - **PyTorch** - Deep learning framework for ML models
            - **PIL (Pillow)** - Image processing and manipulation
            - **NumPy** - Numerical computing and array operations
            - **OpenCV** - Computer vision and image analysis
            - **Scikit-learn** - Machine learning utilities
        """)

        st.markdown("#### Frontend")
        st.markdown("""
            - **Streamlit** - Data app framework for web applications
            - **Plotly** - Interactive charts and visualizations
            - **HTML/CSS** - Custom styling and UI components
        """)

    with col2:
        st.markdown("#### ML & AI")
        st.markdown("""
            - **Convolutional Neural Networks** - Image classification
            - **Transfer Learning** - Pre-trained models for accuracy
            - **Computer Vision** - Leaf disease pattern recognition
            - **Statistical Analysis** - Confidence scoring and probability
        """)

        st.markdown("#### Infrastructure")
        st.markdown("""
            - **RESTful API** - Standard HTTP methods and responses
            - **JSON Data Format** - Structured data exchange
            - **Multipart File Upload** - Image processing
            - **Error Handling** - Comprehensive error responses
            - **Logging** - Application monitoring and debugging
        """)


def show_how_it_works():
    """Show how the system works"""
    st.markdown("---")
    st.markdown("### 🔧 How It Works")

    steps = [
        {
            "step": "1",
            "title": "Image Upload",
            "description": "Upload a clear image of a plant leaf showing disease symptoms through the web interface"
        },
        {
            "step": "2",
            "title": "Image Preprocessing",
            "description": "The system processes and enhances the image to optimize it for disease detection"
        },
        {
            "step": "3",
            "title": "AI Analysis",
            "description": "Advanced ML models analyze the image to identify potential diseases and calculate confidence scores"
        },
        {
            "step": "4",
            "title": "Result Display",
            "description": "Detection results are displayed with disease information, treatment options, and prevention strategies"
        },
        {
            "step": "5",
            "title": "Database Lookup",
            "description": "Comprehensive disease information is retrieved from the database for detailed guidance"
        }
    ]

    for step_info in steps:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"""
                <div style='background-color: #2E7D32; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.2rem; margin-bottom: 1rem;'>
                    {step_info['step']}
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div style='background-color: white; border-radius: 8px; padding: 1rem; border-left: 4px solid #2E7D32; margin-bottom: 1rem;'>
                    <h4 style='color: #2E7D32; margin: 0 0 0.5rem 0;'>{step_info['title']}</h4>
                    <p style='color: #666; margin: 0;'>{step_info['description']}</p>
                </div>
            """, unsafe_allow_html=True)


def show_supported_diseases():
    """Show supported disease information"""
    st.markdown("---")
    st.markdown("### 🌱 Supported Disease Categories")

    categories = [
        {
            "name": "Fungal Diseases",
            "examples": ["Leaf Rust", "Powdery Mildew", "Early Blight", "Late Blight"],
            "icon": "🍄",
            "color": "#4CAF50"
        },
        {
            "name": "Bacterial Diseases",
            "examples": ["Bacterial Leaf Spot", "Bacterial Wilt"],
            "icon": "🦠",
            "color": "#FF9800"
        },
        {
            "name": "Viral Diseases",
            "examples": ["Tobacco Mosaic Virus", "Tomato Yellow Leaf Curl"],
            "icon": "🦠",
            "color": "#F44336"
        },
        {
            "name": "Nutritional Issues",
            "examples": ["Nitrogen Deficiency", "Iron Deficiency"],
            "icon": "🌿",
            "color": "#FFC107"
        },
        {
            "name": "Environmental Stress",
            "examples": ["Heat Stress", "Water Stress", "Light Deficiency"],
            "icon": "☀️",
            "color": "#2196F3"
        },
        {
            "name": "Insect Damage",
            "examples": ["Aphid Damage", "Spider Mite Damage"],
            "icon": "🐛",
            "color": "#9C27B0"
        }
    ]

    # Display category cards
    cols = st.columns(3)
    for i, category in enumerate(categories):
        with cols[i % 3]:
            st.markdown(f"""
                <div style='background-color: white; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #E0E0E0; margin-bottom: 1rem; text-align: center;'>
                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>{category['icon']}</div>
                    <h4 style='color: {category['color']}; margin: 0.5rem 0;'>{category['name']}</h4>
                    <div style='color: #666; font-size: 0.85rem; margin-bottom: 1rem;'>
                        <strong>Examples:</strong> {', '.join(category['examples'])}
                    </div>
                </div>
            """, unsafe_allow_html=True)


def show_api_information():
    """Show API information"""
    st.markdown("---")
    st.markdown("### 🔌 API Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### API Endpoints")
        st.markdown("""
            - **Health Check**: `GET /api/health`
            - **Disease Detection**: `POST /api/detect`
            - **Batch Detection**: `POST /api/detect/batch`
            - **Disease Database**: `GET /api/diseases`
            - **Search Diseases**: `GET /api/diseases/search`
            - **API Documentation**: Available at `/docs`
        """)

    with col2:
        st.markdown("#### Response Format")
        st.markdown("""
            All API responses follow a consistent format:
            ```json
            {
              "success": true,
              "data": {...},
              "error": null,
              "timestamp": "2024-01-15T10:30:00Z"
            }
            ```
        """)

        # API status check
        if st.button("🔄 Check API Status", key="check_api_status"):
            check_api_status()


def check_api_status():
    """Check and display API status"""
    if 'api_client' in st.session_state:
        with st.spinner("Checking API status..."):
            health_response = st.session_state.api_client.health_check(detailed=True)

            if health_response.get('success'):
                st.success("🟢 API is Online and Healthy")

                # Show detailed status
                with st.expander("Detailed Status", expanded=True):
                    status_data = health_response
                    st.metric("Status", status_data.get('status', 'Unknown'))
                    st.metric("Service", status_data.get('service', 'Unknown'))
                    st.metric("Model Loaded", "Yes" if status_data.get('model_loaded') else "No")
                    st.metric("Database Available", "Yes" if status_data.get('database_available') else "No")
            else:
                st.error("🔴 API is Offline or Unhealthy")
                st.warning("Please ensure the backend server is running and accessible.")
    else:
        st.error("🔴 API client not initialized")


def show_system_requirements():
    """Show system requirements"""
    st.markdown("---")
    st.markdown("### 💻 System Requirements")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Minimum Requirements")
        st.markdown("""
            - **Python**: 3.8 or higher
            - **Memory**: 4GB RAM
            - **Storage**: 500MB available space
            - **Network**: Internet connection for API
        """)

    with col2:
        st.markdown("#### Recommended Requirements")
        st.markdown("""
            - **Python**: 3.9 or higher
            - **Memory**: 8GB RAM or more
            - **Storage**: 1GB available space
            - **Network**: Stable internet connection
            - **Display**: 1920x1080 resolution or higher
        """)

    # Browser compatibility
    st.markdown("#### Browser Compatibility")
    st.markdown("""
    - **Chrome**: 90+ (Recommended)
    - **Firefox**: 88+
    - **Safari**: 14+
    - **Edge**: 90+
    - **Mobile**: iOS Safari 14+, Android Chrome 90+
    """)


def show_contact_information():
    """Show contact and support information"""
    st.markdown("---")
    st.markdown("### 📧 Support & Contact")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Getting Help")
        st.markdown("""
        - **Documentation**: Available in this About section
        - **API Docs**: Visit `/docs` endpoint for detailed API documentation
        - **Troubleshooting**: Check the sidebar API status indicator
        - **Issues**: Report bugs or feature requests through project repository
        """)

    with col2:
        st.markdown("#### Development")
        st.markdown("""
        - **Framework**: Built with Streamlit + FastAPI
        - **License**: Open source (check repository for details)
        - **Contributing**: Welcome community contributions
        - **Updates**: Regular updates and improvements planned
        """)

    # Build information
    st.markdown("---")
    st.markdown("#### Build Information")

    build_info = {
        "Version": "1.0.0",
        "Build Date": datetime.now().strftime("%Y-%m-%d"),
        "Python Version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "Environment": "Development"
    }

    for key, value in build_info.items():
        st.metric(key, value)