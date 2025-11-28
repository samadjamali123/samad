"""
Plant Disease Detection System - Main Streamlit Application
"""

import os
import sys
import streamlit as st
import requests
from datetime import datetime
import traceback

# Add backend path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import frontend components and utilities
from utils.api_client import APIClient
from utils.ui_helpers import (
    load_custom_css,
    set_page_config,
    create_sidebar,
    show_error_message,
    show_success_message
)

# Import pages
from pages.home import show_home_page
from pages.results import show_results_page
from pages.diseases import show_diseases_page
from pages.about import show_about_page

# Initialize session state variables
def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'api_client' not in st.session_state:
        st.session_state.api_client = APIClient()

    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'

    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = []

    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

    if 'selected_disease' not in st.session_state:
        st.session_state.selected_disease = None

    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []


def main():
    """Main application entry point"""
    # Configure page settings
    set_page_config()

    # Load custom CSS
    load_custom_css()

    # Initialize session state
    init_session_state()

    # Create sidebar navigation
    selected_page = create_sidebar()

    # Update current page if changed
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page

    # Page routing
    try:
        if st.session_state.current_page == 'Home':
            show_home_page()
        elif st.session_state.current_page == 'Disease Database':
            show_diseases_page()
        elif st.session_state.current_page == 'About':
            show_about_page()
        elif st.session_state.current_page == 'Results' and st.session_state.detection_results:
            show_results_page()
        else:
            # Default to home page if trying to access results without data
            if st.session_state.current_page == 'Results':
                show_error_message("No detection results available. Please upload an image first.")
                st.session_state.current_page = 'Home'
                show_home_page()
            else:
                st.error(f"Unknown page: {st.session_state.current_page}")
                st.session_state.current_page = 'Home'
                show_home_page()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")
        if st.checkbox("Show detailed error"):
            st.code(traceback.format_exc())

    # Footer
    show_footer()


def show_footer():
    """Display application footer"""
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.markdown("🌿 **Plant Disease Detection**")

    with col2:
        st.markdown(
            """
            <div style='text-align: center; color: #666; font-size: 0.9em;'>
            Powered by AI • Built with Streamlit & FastAPI
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        current_time = datetime.now().strftime("%Y")
        st.markdown(f"© {current_time}")


def check_api_connection():
    """Check if backend API is available"""
    try:
        api_client = st.session_state.api_client
        health_status = api_client.health_check()
        return health_status.get('success', False)
    except Exception:
        return False


def show_connection_status():
    """Display API connection status in sidebar"""
    if check_api_connection():
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Disconnected")
        st.warning("""
        The backend API is not available. Please ensure:
        1. The FastAPI server is running on localhost:8000
        2. All dependencies are installed
        3. No firewall is blocking the connection

        To start the backend:
        ```bash
        cd backend
        python main.py
        ```
        """)


if __name__ == "__main__":
    main()