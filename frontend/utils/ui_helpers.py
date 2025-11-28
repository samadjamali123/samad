"""
UI Helper Functions for Plant Disease Detection System
"""

import streamlit as st
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px


def set_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Plant Disease Detection System",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None
    )

    # Custom page title and meta
    st.markdown("""
        <style>
        .stAppHeader {
            background: linear-gradient(90deg, #2E7D32 0%, #388E3C 100%);
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .stAppHeader h1 {
            color: white !important;
            font-weight: 600;
            margin: 0;
        }
        </style>
    """, unsafe_allow_html=True)


def load_custom_css():
    """Load custom CSS for modern UI styling"""
    st.markdown("""
        <style>
        /* Modern Color Scheme */
        :root {
            --primary-color: #2E7D32;
            --secondary-color: #388E3C;
            --accent-color: #FFC107;
            --success-color: #4CAF50;
            --warning-color: #FF9800;
            --error-color: #F44336;
            --background-color: #FAFAFA;
            --text-color: #212121;
            --border-radius: 8px;
            --box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Base Styles */
        .stApp {
            background-color: var(--background-color);
        }

        /* Sidebar Styles */
        .css-1d391kg {
            background-color: white;
            border-right: 1px solid #E0E0E0;
        }

        /* Button Styles */
        .stButton > button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: var(--border-radius);
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: var(--box-shadow);
        }

        .stButton > button:hover {
            background-color: var(--secondary-color);
            transform: translateY(-1px);
        }

        /* Success Button */
        .stButton > button[data-baseweb="primary"] {
            background-color: var(--success-color);
        }

        /* Warning Button */
        .stButton > button[data-baseweb="secondary"] {
            background-color: var(--warning-color);
            color: white;
        }

        /* Error Button */
        .stButton > button[data-baseweb="tertiary"] {
            background-color: var(--error-color);
            color: white;
        }

        /* Card Styles */
        .card {
            background-color: white;
            border-radius: var(--border-radius);
            padding: 1.5rem;
            box-shadow: var(--box-shadow);
            border: 1px solid #E0E0E0;
            margin-bottom: 1rem;
        }

        .card-header {
            background-color: var(--primary-color);
            color: white;
            padding: 0.75rem 1rem;
            border-radius: var(--border-radius) var(--border-radius) 0 0;
            margin: -1.5rem -1.5rem 1rem -1.5rem;
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin: 0;
            color: var(--text-color);
        }

        /* Status Indicators */
        .status-success {
            background-color: var(--success-color);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            display: inline-block;
        }

        .status-warning {
            background-color: var(--warning-color);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            display: inline-block;
        }

        .status-error {
            background-color: var(--error-color);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            display: inline-block;
        }

        /* Progress Bar Styles */
        .stProgress > div > div > div {
            background-color: var(--primary-color);
        }

        /* Metric Cards */
        .metric-card {
            background: white;
            border-radius: var(--border-radius);
            padding: 1.5rem;
            box-shadow: var(--box-shadow);
            text-align: center;
            border: 1px solid #E0E0E0;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary-color);
        }

        .metric-label {
            font-size: 0.875rem;
            color: #666;
            margin-top: 0.5rem;
        }

        /* Disease Severity Colors */
        .severity-low {
            color: var(--success-color);
            background-color: rgba(76, 175, 80, 0.1);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
        }

        .severity-moderate {
            color: var(--warning-color);
            background-color: rgba(255, 152, 0, 0.1);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
        }

        .severity-high {
            color: #FF5722;
            background-color: rgba(255, 87, 34, 0.1);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
        }

        .severity-critical {
            color: var(--error-color);
            background-color: rgba(244, 67, 54, 0.1);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
        }

        /* Confidence Score Visualization */
        .confidence-bar {
            background-color: #E0E0E0;
            border-radius: 10px;
            overflow: hidden;
            height: 8px;
            margin: 0.5rem 0;
        }

        .confidence-fill {
            height: 100%;
            transition: width 0.5s ease;
            border-radius: 10px;
        }

        .confidence-high {
            background-color: var(--success-color);
        }

        .confidence-medium {
            background-color: var(--warning-color);
        }

        .confidence-low {
            background-color: var(--error-color);
        }

        /* Image Upload Area */
        .upload-area {
            border: 2px dashed var(--primary-color);
            border-radius: var(--border-radius);
            padding: 2rem;
            text-align: center;
            background-color: rgba(46, 125, 50, 0.05);
            transition: all 0.3s ease;
        }

        .upload-area:hover {
            border-color: var(--secondary-color);
            background-color: rgba(46, 125, 50, 0.1);
        }

        /* Result Display */
        .result-container {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 2rem;
            align-items: start;
        }

        @media (max-width: 768px) {
            .result-container {
                grid-template-columns: 1fr;
                gap: 1rem;
            }
        }

        /* Loading Animation */
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid var(--primary-color);
            border-radius: 50%;
            border-top-color: transparent;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Footer Styles */
        .footer {
            text-align: center;
            padding: 2rem 0;
            color: #666;
            border-top: 1px solid #E0E0E0;
            margin-top: 3rem;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-color);
        }

        .main {
            color: var(--text-color);
        }

        /* Form Styles */
        .stSelectbox > div > div {
            background-color: white;
            border-radius: var(--border-radius);
        }

        .stTextInput > div > div > input {
            border-radius: var(--border-radius);
            border: 1px solid #E0E0E0;
        }

        /* Table Styles */
        .stTable {
            background-color: white;
            border-radius: var(--border-radius);
            overflow: hidden;
        }

        /* Success/Warning/Error Messages */
        .success-message {
            background-color: var(--success-color);
            color: white;
            padding: 1rem;
            border-radius: var(--border-radius);
            margin: 1rem 0;
        }

        .warning-message {
            background-color: var(--warning-color);
            color: white;
            padding: 1rem;
            border-radius: var(--border-radius);
            margin: 1rem 0;
        }

        .error-message {
            background-color: var(--error-color);
            color: white;
            padding: 1rem;
            border-radius: var(--border-radius);
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


def create_sidebar() -> str:
    """Create sidebar navigation and return selected page"""
    with st.sidebar:
        # Header
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2E7D32; margin: 0;'>🌿</h1>
                <h3 style='color: #2E7D32; margin: 0.5rem 0;'>Plant Disease<br>Detection</h3>
            </div>
        """, unsafe_allow_html=True)

        # Navigation
        st.markdown("### Navigation")
        page = st.selectbox(
            "Choose a page:",
            ['Home', 'Disease Database', 'About'],
            key='nav_select',
            help="Navigate through different sections of the application"
        )

        # Add Results page if we have detection results
        if 'detection_results' in st.session_state and st.session_state.detection_results:
            if "Results" not in ['Home', 'Disease Database', 'About']:
                pages_with_results = ['Home', 'Disease Database', 'About', 'Results']
                page_index = pages_with_results.index(page)
                page = st.selectbox(
                    "Choose a page:",
                    pages_with_results,
                    index=page_index,
                    key='nav_select_with_results'
                )
            else:
                # Create a separate selectbox that includes Results
                all_pages = ['Home', 'Disease Database', 'About', 'Results']
                if page == 'Results':
                    current_index = 3
                else:
                    current_index = ['Home', 'Disease Database', 'About'].index(page)
                page = st.selectbox(
                    "Choose a page:",
                    all_pages,
                    index=current_index,
                    key='nav_with_results'
                )

        # Separator
        st.markdown("---")

        # API Status
        st.markdown("### API Status")
        if 'api_client' in st.session_state:
            with st.spinner("Checking API status..."):
                health_status = st.session_state.api_client.health_check()

                if health_status.get('success'):
                    st.success("🟢 API Connected")
                    st.caption(f"Version: {health_status.get('service', 'Unknown')}")
                else:
                    st.error("🔴 API Disconnected")
                    st.warning("Please start the backend server")
                    with st.expander("Troubleshooting"):
                        st.code("""
# Start backend server
cd /workspace/cmiijrwzj01gxtmim3iqqopla/samad/backend
python main.py
                        """)
        else:
            st.error("🔴 API Client Not Initialized")

        # Quick Stats
        if 'detection_history' in st.session_state:
            st.markdown("### Quick Stats")
            history_count = len(st.session_state.detection_history)
            st.metric("Analyses Today", history_count)

            if history_count > 0:
                last_analysis = st.session_state.detection_history[-1]['timestamp']
                st.caption(f"Last: {last_analysis}")

        # Settings
        st.markdown("---")
        st.markdown("### Settings")

        # Confidence threshold
        if 'confidence_threshold' not in st.session_state:
            st.session_state.confidence_threshold = 0.6

        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.confidence_threshold,
            step=0.1,
            help="Minimum confidence level for disease detection"
        )
        st.session_state.confidence_threshold = confidence

        # Theme selection (placeholder)
        theme = st.selectbox(
            "Theme",
            ["Light", "Dark"],
            index=0,
            help="Choose application theme"
        )

        # About section
        st.markdown("---")
        with st.expander("About"):
            st.markdown("""
            **Plant Disease Detection System**

            Version: 1.0.0
            Built with: Streamlit + FastAPI

            *Upload plant leaf images for AI-powered disease detection*
            """)

        return page


def show_error_message(message: str, details: Optional[str] = None):
    """Display error message with optional details"""
    st.markdown(f"""
        <div class='error-message'>
            <strong>Error:</strong> {message}
            {f"<br><details><summary>Details</summary><code>{details}</code></details>" if details else ""}
        </div>
    """, unsafe_allow_html=True)


def show_success_message(message: str):
    """Display success message"""
    st.markdown(f"""
        <div class='success-message'>
            <strong>Success!</strong> {message}
        </div>
    """, unsafe_allow_html=True)


def show_warning_message(message: str):
    """Display warning message"""
    st.markdown(f"""
        <div class='warning-message'>
            <strong>Warning:</strong> {message}
        </div>
    """, unsafe_allow_html=True)


def create_metric_card(title: str, value: str, subtitle: Optional[str] = None):
    """Create a styled metric card"""
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{value}</div>
            <div class='metric-label'>{title}</div>
            {f"<div style='font-size: 0.75rem; color: #666; margin-top: 0.5rem;'>{subtitle}</div>" if subtitle else ""}
        </div>
    """, unsafe_allow_html=True)


def create_severity_badge(severity: str) -> str:
    """Create HTML for severity badge"""
    severity_class = severity.lower().replace(" ", "-")
    return f"""
        <span class='severity-{severity_class}'>
            {severity}
        </span>
    """


def create_confidence_bar(confidence: float) -> str:
    """Create HTML for confidence visualization"""
    confidence_percentage = confidence * 100

    if confidence >= 0.8:
        fill_class = "confidence-high"
    elif confidence >= 0.6:
        fill_class = "confidence-medium"
    else:
        fill_class = "confidence-low"

    return f"""
        <div class='confidence-bar'>
            <div class='confidence-fill {fill_class}' style='width: {confidence_percentage}%;'></div>
        </div>
        <small>Confidence: {confidence_percentage:.1f}%</small>
    """


def create_loading_spinner(text: str = "Processing...") -> str:
    """Create custom loading spinner"""
    return f"""
        <div style='display: flex; align-items: center; gap: 0.5rem;'>
            <div class='loading-spinner'></div>
            <span>{text}</span>
        </div>
    """


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_processing_time(seconds: float) -> str:
    """Format processing time in human readable format"""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    else:
        return f"{seconds:.2f} seconds"


def create_disease_card(disease: Dict[str, Any]) -> str:
    """Create HTML card for disease display"""
    severity_badge = create_severity_badge(disease.get('severity', 'Unknown'))

    symptoms = disease.get('symptoms', [])
    symptoms_preview = symptoms[0] if symptoms else "No symptoms available"

    affected_plants = disease.get('affected_plants', [])
    plants_preview = ", ".join(affected_plants[:3])
    if len(affected_plants) > 3:
        plants_preview += f" +{len(affected_plants) - 3} more"

    return f"""
        <div class='card'>
            <div class='card-header'>
                <h3 style='margin: 0; color: white;'>{disease.get('name', 'Unknown Disease')}</h3>
                <div style='margin-top: 0.5rem;'>{severity_badge}</div>
            </div>
            <p><strong>Scientific Name:</strong> {disease.get('scientific_name', 'N/A')}</p>
            <p><strong>Category:</strong> {disease.get('category', 'Unknown')}</p>
            <p><strong>Affected Plants:</strong> {plants_preview}</p>
            <p><strong>Key Symptom:</strong> {symptoms_preview}</p>
            {f"<p><strong>Contagious:</strong> {'Yes' if disease.get('contagious') else 'No'}</p>" if 'contagious' in disease else ""}
        </div>
    """


def create_progress_bar_with_text(progress: float, text: str) -> str:
    """Create progress bar with custom text"""
    return f"""
        <div style='display: flex; align-items: center; gap: 1rem;'>
            <div style='flex: 1;'>
                <div style='background-color: #E0E0E0; border-radius: 10px; overflow: hidden; height: 8px;'>
                    <div style='width: {progress * 100}%; height: 100%; background-color: #2E7D32; transition: width 0.3s ease;'></div>
                </div>
            </div>
            <div style='font-size: 0.875rem; color: #666; min-width: 100px; text-align: right;'>
                {text}
            </div>
        </div>
    """


def show_page_header(title: str, subtitle: Optional[str] = None):
    """Show consistent page header"""
    st.markdown(f"""
        <div style='background: linear-gradient(90deg, #2E7D32 0%, #388E3C 100%); color: white; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; text-align: center;'>
            <h1 style='margin: 0; font-size: 2rem;'>{title}</h1>
            {f"<p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>{subtitle}</p>" if subtitle else ""}
        </div>
    """, unsafe_allow_html=True)


def show_empty_state(
    icon: str,
    title: str,
    description: str,
    action_text: Optional[str] = None,
    action_callback: Optional[callable] = None
):
    """Show empty state with optional action"""
    st.markdown(f"""
        <div style='text-align: center; padding: 3rem 0; color: #666;'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>{icon}</div>
            <h3 style='color: #212121; margin-bottom: 1rem;'>{title}</h3>
            <p style='margin-bottom: 2rem;'>{description}</p>
            {f"<div style='margin-top: 1rem;'>{action_text}</div>" if action_text else ""}
        </div>
    """, unsafe_allow_html=True)

    if action_callback and action_text:
        if st.button(action_text, key="empty_state_action"):
            action_callback()


def create_download_button(
    data: str,
    filename: str,
    button_text: str = "Download",
    mime_type: str = "text/plain"
):
    """Create styled download button"""
    b64 = base64.b64encode(data.encode()).decode()
    href = f"<a href='data:{mime_type};base64,{b64}' download='{filename}'>{button_text}</a>"
    return href


def show_confidence_gauge(confidence: float, title: str = "Confidence Score"):
    """Show confidence score as a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 100]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#2E7D32"},
            'steps': [
                {'range': [0, 50], 'color': "#F44336"},
                {'range': [50, 80], 'color': "#FF9800"},
                {'range': [80, 100], 'color': "#4CAF50"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))

    fig.update_layout(
        height=300,
        font={'color': "#212121"},
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


def create_disease_severity_chart(diseases: List[Dict[str, Any]]):
    """Create a chart showing disease severity distribution"""
    if not diseases:
        return

    severity_counts = {}
    for disease in diseases:
        severity = disease.get('severity', 'Unknown')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    # Define color mapping
    color_map = {
        'Low': '#4CAF50',
        'Moderate': '#FF9800',
        'High': '#FF5722',
        'Critical': '#F44336'
    }

    colors = [color_map.get(sev, '#666666') for sev in severity_counts.keys()]

    fig = px.pie(
        values=list(severity_counts.values()),
        names=list(severity_counts.keys()),
        title="Disease Severity Distribution",
        color_discrete_map=color_map
    )

    fig.update_layout(
        height=400,
        font=dict(color="#212121"),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)


def create_timeline_chart(data_points: List[Dict[str, Any]]):
    """Create a timeline chart for detection history"""
    if not data_points:
        return

    # Prepare data
    timestamps = []
    confidence_scores = []

    for point in data_points:
        timestamp = point.get('timestamp', '')
        confidence = point.get('confidence', 0)

        timestamps.append(timestamp)
        confidence_scores.append(confidence * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=confidence_scores,
        mode='lines+markers',
        name='Confidence Score',
        line=dict(color='#2E7D32', width=3),
        marker=dict(size=8, color='#2E7D32')
    ))

    fig.update_layout(
        title="Detection Confidence Over Time",
        xaxis_title="Time",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        height=300,
        font=dict(color="#212121")
    )

    st.plotly_chart(fig, use_container_width=True)


def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."


def create_search_filters(
    categories: List[str],
    severity_levels: List[str],
    affected_plants: List[str]
) -> Dict[str, Any]:
    """Create search filter widgets"""
    with st.expander("Search Filters", expanded=False):
        category_filter = st.multiselect(
            "Categories",
            categories,
            help="Filter by disease category"
        )

        severity_filter = st.multiselect(
            "Severity Levels",
            severity_levels,
            help="Filter by severity level"
        )

        plant_filter = st.multiselect(
            "Affected Plants",
            affected_plants,
            help="Filter by affected plant type"
        )

        contagious_filter = st.selectbox(
            "Contagious",
            ["All", "Contagious Only", "Non-Contagious Only"],
            help="Filter by contagious status"
        )

        return {
            'categories': category_filter,
            'severity': severity_filter,
            'plants': plant_filter,
            'contagious': contagious_filter
        }


def apply_filters_to_diseases(
    diseases: List[Dict[str, Any]],
    filters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Apply filters to disease list"""
    filtered_diseases = []

    for disease in diseases:
        # Category filter
        if filters['categories']:
            if disease.get('category') not in filters['categories']:
                continue

        # Severity filter
        if filters['severity']:
            if disease.get('severity') not in filters['severity']:
                continue

        # Plants filter
        if filters['plants']:
            disease_plants = set(disease.get('affected_plants', []))
            filter_plants = set(filters['plants'])
            if not disease_plants.intersection(filter_plants):
                continue

        # Contagious filter
        if filters['contagious'] != 'All':
            disease_contagious = disease.get('contagious', False)
            if filters['contagious'] == 'Contagious Only' and not disease_contagious:
                continue
            elif filters['contagious'] == 'Non-Contagious Only' and disease_contagious:
                continue

        filtered_diseases.append(disease)

    return filtered_diseases