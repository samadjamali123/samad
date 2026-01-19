"""
Plant Leaf Disease Detector - Streamlit Frontend

Main application file for the Streamlit-based frontend interface.
Provides a user-friendly interface for uploading plant leaf images,
receiving AI analysis results, and viewing comprehensive treatment recommendations.
"""

import streamlit as st
import streamlit.components.v1 as components
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import io
import base64
from datetime import datetime

# Add backend to path for imports (for local development)
backend_path = Path(__file__).parent.parent / "backend"
if backend_path.exists():
    sys.path.insert(0, str(backend_path))

# Import frontend components
from frontend.components.image_uploader import ImageUploader
from frontend.components.results_display import ResultsDisplay
from frontend.components.treatment_card import TreatmentCard
from frontend.components.camera_component import CameraComponent
from frontend.utils.api_client import APIClient
from frontend.utils.image_processing import ImageProcessor as FrontendImageProcessor
from frontend.utils.styling import setup_custom_css
from frontend.utils.camera_utils import create_camera_container

# Configuration
API_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
APP_TITLE = "🌱 Plant Leaf Disease Detector"
APP_DESCRIPTION = """
Upload a photo of a plant leaf to instantly identify the plant type, detect any diseases present,
and get comprehensive treatment recommendations from agricultural experts.
"""

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Plant Disease Detector",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': """
        # Plant Leaf Disease Detector

        **Version:** 1.0.0

        An AI-powered system that identifies plant diseases from leaf images using advanced machine learning models.
        Provides detailed information about symptoms, causes, and comprehensive treatment recommendations.

        **Features:**
        - 🔍 Multi-model AI analysis
        - 🌿 Comprehensive plant database
        - 💊 Organic and chemical treatments
        - 📸 Visual disease references
        - 📱 Mobile-friendly interface

        **Support:**
        - 🍅 50+ major crops covered
        - 🎯 20+ most damaging diseases per crop
        - 🌍 Global agricultural databases
            """
        }
    )

def create_session_state():
    """Initialize and manage session state variables"""
    # Initialize session state if not exists
    if 'api_client' not in st.session_state:
        st.session_state.api_client = APIClient(API_BASE_URL)

    if 'current_image' not in st.session_state:
        st.session_state.current_image = None

    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    if 'selected_disease' not in st.session_state:
        st.session_state.selected_disease = None

    if 'page_history' not in st.session_state:
        st.session_state.page_history = ['home']

    if 'theme_preference' not in st.session_state:
        st.session_state.theme_preference = 'auto'

def display_header():
    """Display application header with title and description"""
    st.markdown("""
    <div class="main-header">
        <div class="header-content">
            <h1>🌱 Plant Leaf Disease Detector</h1>
            <p class="header-description">
                Upload a photo of a plant leaf to instantly identify the plant type,
                detect any diseases present, and get comprehensive treatment recommendations.
            </p>
        </div>
        <div class="header-stats">
            <div class="stat-item">
                <span class="stat-number">50+</span>
                <span class="stat-label">Crops Supported</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">1000+</span>
                <span class="stat-label">Diseases Covered</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">95%</span>
                <span class="stat-label">Accuracy Rate</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Create sidebar with navigation and information"""
    with st.sidebar:
        st.markdown("### 🌿 Navigation")

        # Navigation buttons
        if st.button("📸 New Analysis", type="primary", use_container_width=True):
            st.session_state.page_history = ['home']
            st.session_state.current_image = None
            st.session_state.analysis_results = None
            st.session_state.selected_disease = None
            st.rerun()

        if st.session_state.analysis_results:
            if st.button("📊 View Results", use_container_width=True):
                st.session_state.page_history = ['home', 'results']
                st.rerun()

        if st.session_state.selected_disease:
            if st.button("💊 View Treatments", use_container_width=True):
                st.session_state.page_history = ['home', 'results', 'treatments']
                st.rerun()

        st.markdown("---")

        # Quick info section
        st.markdown("### 📋 Quick Tips")

        tips = [
            "📸 Take clear, well-lit photos",
            "🌿 Include multiple leaves if possible",
            "☀️ Avoid shadows and glare",
            "🔍 Focus on affected areas",
            "📏 Fill at least 50% of the frame"
        ]

        for tip in tips:
            st.markdown(f"<div class='tip-item'>{tip}</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Health check status
        st.markdown("### 🏥 System Status")

        try:
            health_response = st.session_state.api_client.get_health()
            if health_response.get('success', False):
                st.success("🟢 All systems operational")

                # Display model status
                services = health_response.get('services', {})
                for service_name, status in services.items():
                    if status:
                        st.success(f"✅ {service_name.replace('_', ' ').title()}")
                    else:
                        st.warning(f"⚠️ {service_name.replace('_', ' ').title()}")
            else:
                st.error("🔴 Service issues detected")
        except Exception as e:
            st.error(f"🔴 Cannot check system status")

        st.markdown("---")

        # Settings
        st.markdown("### ⚙️ Settings")

        # Theme selector
        theme = st.selectbox(
            "Theme",
            options=["Auto", "Light", "Dark"],
            index=["auto", "light", "dark"].index(st.session_state.theme_preference)
        )
        st.session_state.theme_preference = theme.lower()

        # Language selector (placeholder for future multi-language support)
        language = st.selectbox(
            "Language",
            options=["English", "Spanish", "French", "Hindi", "Mandarin"],
            index=0
        )

        # About section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Plant Disease Detector v1.0.0**

            This AI-powered tool helps farmers and gardeners identify plant diseases
            from leaf photos using advanced machine learning models.

            **How it works:**
            1. Upload a clear leaf photo
            2. AI analyzes for plant identification and disease detection
            3. Get comprehensive treatment recommendations

            **Data Privacy:** Your images are processed securely and are not stored permanently.
            """)

def handle_camera_capture(image_data):
    """Handle camera capture event"""
    if image_data:
        st.session_state.camera_captured = True
        st.session_state.current_image = image_data
        st.success("📸 Image captured successfully! Analyzing...")
        return True
    return False


def display_home_page():
    """Display main home page with image upload and camera integration"""
    # Create two columns for main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📸 Upload Plant Leaf Image")

        # Create image uploader component
        uploader = ImageUploader(st.session_state.api_client)

        # Process uploaded image
        if uploader.uploaded_file is not None:
            st.session_state.current_image = uploader.uploaded_file

            # Show loading state while analyzing
            with st.spinner("🔍 Analyzing image with AI models..."):
                try:
                    results = st.session_state.api_client.analyze_image(uploader.uploaded_file)
                    st.session_state.analysis_results = results

                    if results.get('success', False):
                        st.success("✅ Analysis completed successfully!")
                        st.session_state.page_history = ['home', 'results']
                        st.rerun()
                    else:
                        st.error(f"❌ Analysis failed: {results.get('error_message', 'Unknown error')}")

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.markdown("""
                    **Troubleshooting:**
                    - Check your internet connection
                    - Ensure the image is clear and well-lit
                    - Try uploading a different image
                    - Contact support if the issue persists
                    """)

    with col2:
        st.markdown("### 📖 What We Can Detect")

        st.markdown("""
        <div class="feature-list">
            <div class="feature-item">
                <div class="feature-icon">🍅</div>
                <div class="feature-text">
                    <strong>50+ Crops</strong><br>
                    Vegetables, fruits, grains, legumes, and more
                </div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🔬</div>
                <div class="feature-text">
                    <strong>1000+ Diseases</strong><br>
                    Common and economically damaging plant diseases
                </div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">💊</div>
                <div class="feature-text">
                    <strong>Smart Treatments</strong><br>
                    Organic, chemical, and preventive measures
                </div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">📊</div>
                <div class="feature-text">
                    <strong>Confidence Scores</strong><br>
                    AI confidence levels for reliable results
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🎯 Success Stories")

        stories = [
            {
                "crop": "🍅 Tomato",
                "issue": "Early Blight detected",
                "result": "Saved 80% of crop with early treatment"
            },
            {
                "crop": "🥔 Potato",
                "issue": "Late Blight identified",
                "result": "Prevented farm-wide outbreak"
            },
            {
                "crop": "🍎 Apple",
                "issue": "Fire Blight spotted",
                "result": "Applied targeted organic treatment"
            }
        ]

        for story in stories:
            st.markdown(f"""
            <div class="story-card">
                <div class="story-crop">{story['crop']}</div>
                <div class="story-issue">{story['issue']}</div>
                <div class="story-result">✅ {story['result']}</div>
            </div>
            """, unsafe_allow_html=True)

def display_results_page():
    """Display analysis results page"""
    if not st.session_state.analysis_results:
        st.warning("No analysis results available. Please upload an image first.")
        return

    st.markdown("### 📊 Analysis Results")

    # Display results component
    results_display = ResultsDisplay(
        st.session_state.analysis_results,
        st.session_state.current_image
    )

    # Add Grad-CAM section if image is available
    if st.session_state.current_image:
        grad_cam_extension = results_display.GradCAMExtension(
            st.session_state.analysis_results,
            st.session_state.current_image
        )
        grad_cam_extension.display_grad_cam_section()

    # Get selected disease from results
    if results_display.selected_disease:
        st.session_state.selected_disease = results_display.selected_disease

        # Show treatment recommendation button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💊 View Treatment Recommendations", type="primary", use_container_width=True):
                st.session_state.page_history = ['home', 'results', 'treatments']
                st.rerun()

def display_treatments_page():
    """Display treatment recommendations page"""
    if not st.session_state.selected_disease:
        st.warning("No disease selected. Please analyze an image first.")
        return

    st.markdown(f"### 💊 Treatment Recommendations")

    # Display treatment card component
    treatment_card = TreatmentCard(
        st.session_state.selected_disease,
        st.session_state.api_client
    )

def display_footer():
    """Display application footer"""
    st.markdown("""
    <div class="app-footer">
        <div class="footer-content">
            <div class="footer-section">
                <h4>🌱 About</h4>
                <p>AI-powered plant disease detection helping farmers protect their crops worldwide.</p>
            </div>
            <div class="footer-section">
                <h4>🔗 Resources</h4>
                <ul>
                    <li><a href="#" onclick="alert('Documentation coming soon!')">Documentation</a></li>
                    <li><a href="#" onclick="alert('API docs available at /docs')">API</a></li>
                    <li><a href="#" onclick="alert('Research papers coming soon!')">Research</a></li>
                </ul>
            </div>
            <div class="footer-section">
                <h4>📞 Support</h4>
                <ul>
                    <li><a href="#" onclick="alert('Contact support@plantdisease.ai')">Email Support</a></li>
                    <li><a href="#" onclick="alert('Community forum coming soon!')">Community</a></li>
                    <li><a href="#" onclick="alert('Bug reporting coming soon!')">Report Issue</a></li>
                </ul>
            </div>
        </div>
        <div class="footer-bottom">
            <p>&copy; 2024 Plant Disease Detector. Built with ❤️ for farmers worldwide.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def register_pwa_features():
    """Register PWA manifest, service worker, and enhanced features"""
    # PWA manifest and service worker registration with enhanced theming
    pwa_registration = """
    <!-- PWA Manifest and Service Worker Registration -->
    <link rel="manifest" href="/static/manifest.json">
    <meta name="theme-color" content="#4CAF50">
    <link rel="apple-touch-icon" href="/static/icons/icon-192x192.png">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="apple-mobile-web-app-title" content="PlantAI">
    <meta name="application-name" content="PlantAI">
    <meta name="msapplication-TileColor" content="#4CAF50">
    <meta name="msapplication-config" content="/static/browserconfig.xml">

    <!-- Enhanced PWA Features -->
    <meta name="description" content="AI-powered plant disease detection and treatment recommendations">
    <meta name="keywords" content="plants, disease detection, agriculture, AI, machine learning, farming">
    <meta name="author" content="PlantAI Team">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">

    <!-- Social Media Meta Tags -->
    <meta property="og:title" content="PlantAI - Disease Detection">
    <meta property="og:description" content="AI-powered plant disease detection and treatment recommendations">
    <meta property="og:image" content="/static/icons/icon-512x512.png">
    <meta property="og:url" content="/">
    <meta property="og:type" content="website">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="PlantAI - Disease Detection">
    <meta name="twitter:description" content="AI-powered plant disease detection">

    <!-- Enhanced PWA Features -->
    <style>
        /* PWA Install Prompt */
        .pwa-install-prompt {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(76, 175, 80, 0.3);
            z-index: 1000;
            max-width: 300px;
            animation: slideInUp 0.5s ease-out;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        @keyframes slideInUp {
            from {
                transform: translateY(100px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }

        /* PWA Offline Indicator */
        .offline-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #ff9800;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 600;
            z-index: 1001;
            animation: pulse 2s infinite;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .pwa-install-prompt {
                bottom: 10px;
                right: 10px;
                left: 10px;
                max-width: none;
            }

            .offline-indicator {
                top: 10px;
                right: 10px;
                left: 10px;
                text-align: center;
            }
        }

        /* Enhanced scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
        }

        ::-webkit-scrollbar-thumb {
            background: #4CAF50;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #45a049;
        }

        /* PWA Safe Area insets */
        @supports (padding: max(0px)) {
            .pwa-install-prompt {
                padding: max(1rem, env(safe-area-inset-bottom) + 1rem);
                margin-bottom: env(safe-area-inset-bottom);
            }
        }

        /* Camera component PWA enhancements */
        .camera-container {
            max-width: 100%;
            margin: 1rem 0;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }

        /* PWA Status Bar styling */
        @media (display-mode: standalone) {
            .stApp {
                padding-top: env(safe-area-inset-top);
            }
        }

        /* PWA Splash Screen Styles */
        .splash-screen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            color: white;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .splash-icon {
            width: 120px;
            height: 120px;
            margin-bottom: 2rem;
            animation: splashPulse 2s ease-in-out infinite;
        }

        @keyframes splashPulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.05); opacity: 0.8; }
        }

        .splash-text {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .splash-subtitle {
            font-size: 1rem;
            opacity: 0.8;
        }
    </style>

    <!-- Service Worker Registration with Enhanced Features -->
    <script>
        // PWA Installation Detection
        let isPWA = window.matchMedia('(display-mode: standalone)').matches;
        let isInstalled = localStorage.getItem('pwa-installed') === 'true';

        console.log('PWA Status:', { standalone: isPWA, installed: isInstalled });

        // Service Worker Registration
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', function() {
                navigator.serviceWorker.register('/static/sw.js')
                    .then(function(registration) {
                        console.log('ServiceWorker registration successful with scope: ', registration.scope);

                        // Check for service worker updates
                        registration.addEventListener('updatefound', function() {
                            const newWorker = registration.installing;
                            console.log('New service worker found');

                            newWorker.addEventListener('statechange', function() {
                                if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                                    console.log('Service worker updated');
                                    // Show update notification
                                    showUpdateNotification();
                                }
                            });
                        });
                    })
                    .catch(function(error) {
                        console.log('ServiceWorker registration failed: ', error);
                    });
            });
        }

        // Enhanced PWA Install Prompt
        let deferredPrompt;
        let installPromptShown = localStorage.getItem('install-prompt-shown') === 'true';

        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;

            // Show install prompt after user interaction
            if (!installPromptShown && !isPWA) {
                setTimeout(() => {
                    showInstallPrompt();
                }, 30000); // Show after 30 seconds
            }
        });

        // Track PWA Installation
        window.addEventListener('appinstalled', (evt) => {
            console.log('PWA was installed');
            localStorage.setItem('pwa-installed', 'true');

            // Track installation event
            if (typeof gtag !== 'undefined') {
                gtag('event', 'pwa_install', {
                    'event_category': 'engagement',
                    'event_label': 'pwa_install_success'
                });
            }

            // Hide install prompt if shown
            const prompt = document.querySelector('.pwa-install-prompt');
            if (prompt) {
                prompt.remove();
            }
        });

        function showInstallPrompt() {
            if (document.querySelector('.pwa-install-prompt') || installPromptShown) return;

            const installDiv = document.createElement('div');
            installDiv.className = 'pwa-install-prompt';
            installDiv.innerHTML = `
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <strong>📱 Install PlantAI</strong><br>
                        <small style="opacity: 0.9;">Get our app for faster access and offline support</small>
                    </div>
                    <div style="margin-left: 1rem; display: flex; gap: 0.5rem;">
                        <button onclick="installApp()" style="background: white; color: #4CAF50; border: none; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600; cursor: pointer;">Install</button>
                        <button onclick="dismissPrompt()" style="background: transparent; color: white; border: 1px solid white; padding: 0.5rem; border-radius: 6px; cursor: pointer;">×</button>
                    </div>
                </div>
            `;
            document.body.appendChild(installDiv);
            localStorage.setItem('install-prompt-shown', 'true');
        }

        function installApp() {
            if (deferredPrompt) {
                deferredPrompt.prompt();
                deferredPrompt.userChoice.then((choiceResult) => {
                    if (choiceResult.outcome === 'accepted') {
                        console.log('User accepted install prompt');
                        localStorage.setItem('pwa-installed', 'true');

                        // Track installation event
                        if (typeof gtag !== 'undefined') {
                            gtag('event', 'pwa_install', {
                                'event_category': 'engagement',
                                'event_label': 'pwa_install_accepted'
                            });
                        }
                    } else {
                        console.log('User dismissed install prompt');

                        // Track dismissal event
                        if (typeof gtag !== 'undefined') {
                            gtag('event', 'pwa_install_dismiss', {
                                'event_category': 'engagement',
                                'event_label': 'pwa_install_dismissed'
                            });
                        }
                    }
                    deferredPrompt = null;
                    document.querySelector('.pwa-install-prompt')?.remove();
                });
            }
        }

        function dismissPrompt() {
            const prompt = document.querySelector('.pwa-install-prompt');
            if (prompt) {
                prompt.remove();
            }

            // Track dismissal event
            if (typeof gtag !== 'undefined') {
                gtag('event', 'pwa_install_dismiss', {
                    'event_category': 'engagement',
                    'event_label': 'pwa_install_manual_dismiss'
                });
            }
        }

        function showUpdateNotification() {
            const updateDiv = document.createElement('div');
            updateDiv.className = 'pwa-install-prompt';
            updateDiv.style.background = '#2196F3';
            updateDiv.innerHTML = `
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <strong>🔄 Update Available</strong><br>
                        <small style="opacity: 0.9;">A new version of PlantAI is ready</small>
                    </div>
                    <div style="margin-left: 1rem; display: flex; gap: 0.5rem;">
                        <button onclick="updateApp()" style="background: white; color: #2196F3; border: none; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600; cursor: pointer;">Update</button>
                        <button onclick="dismissUpdate()" style="background: transparent; color: white; border: 1px solid white; padding: 0.5rem; border-radius: 6px; cursor: pointer;">×</button>
                    </div>
                </div>
            `;
            document.body.appendChild(updateDiv);
        }

        function updateApp() {
            window.location.reload();
        }

        function dismissUpdate() {
            document.querySelector('.pwa-install-prompt')?.remove();
        }

        // Enhanced Online/Offline Status Detection
        function updateOnlineStatus() {
            const offlineIndicator = document.querySelector('.offline-indicator');

            if (!navigator.onLine) {
                if (!offlineIndicator) {
                    const indicator = document.createElement('div');
                    indicator.className = 'offline-indicator';
                    indicator.textContent = '🔴 You are offline - Some features may be limited';
                    document.body.appendChild(indicator);
                }

                // Show offline mode notification
                console.log('App is in offline mode');

                // Track offline event
                if (typeof gtag !== 'undefined') {
                    gtag('event', 'offline_mode', {
                        'event_category': 'system',
                        'event_label': 'user_went_offline'
                    });
                }
            } else {
                if (offlineIndicator) {
                    offlineIndicator.remove();
                }

                console.log('App is online');

                // Track online event
                if (typeof gtag !== 'undefined') {
                    gtag('event', 'online_mode', {
                        'event_category': 'system',
                        'event_label': 'user_came_online'
                    });
                }
            }
        }

        // Listen for online/offline events
        window.addEventListener('online', updateOnlineStatus);
        window.addEventListener('offline', updateOnlineStatus);

        // Initial status check
        updateOnlineStatus();

        // PWA Theme Detection and Synchronization
        function synchronizeTheme() {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            const storedTheme = localStorage.getItem('theme-preference');

            // Send theme preference to Streamlit
            if (window.parent && window.parent.postMessage) {
                window.parent.postMessage({
                    type: 'theme_update',
                    theme: storedTheme || (prefersDark ? 'dark' : 'light')
                }, '*');
            }

            console.log('Theme synchronized:', storedTheme || (prefersDark ? 'dark' : 'light'));
        }

        // Initialize theme sync
        synchronizeTheme();

        // Listen for system theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', synchronizeTheme);

        // Enhanced Mobile Capabilities Detection
        function detectMobileCapabilities() {
            const userAgent = navigator.userAgent.toLowerCase();
            const isMobile = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(userAgent);
            const isPWA = window.matchMedia('(display-mode: standalone)').matches;
            const isIOS = /iphone|ipad|ipod/i.test(userAgent);

            if (isMobile) {
                document.body.classList.add('mobile-device');

                if (isPWA) {
                    document.body.classList.add('pwa-mode');
                    console.log('Running as PWA on mobile device');

                    // PWA-specific mobile optimizations
                    if (isIOS) {
                        document.body.classList.add('ios-pwa');
                    }
                }

                // Mobile-specific features
                setupMobileOptimizations();
            }

            // Store capabilities for later use
            localStorage.setItem('device-capabilities', JSON.stringify({
                isMobile,
                isPWA,
                isIOS,
                userAgent: navigator.userAgent
            }));
        }

        function setupMobileOptimizations() {
            // Enable touch-friendly interactions
            document.addEventListener('touchstart', function() {}, false);

            // Optimize viewport for mobile
            const viewport = document.querySelector('meta[name="viewport"]');
            if (viewport && !viewport.content.includes('viewport-fit')) {
                viewport.content += ', viewport-fit=cover';
            }

            console.log('Mobile optimizations applied');
        }

        // Initialize mobile detection
        detectMobileCapabilities();

        // PWA Performance Monitoring
        function monitorPerformance() {
            if ('performance' in window) {
                window.addEventListener('load', function() {
                    setTimeout(function() {
                        const perfData = performance.getEntriesByType('navigation')[0];

                        if (perfData) {
                            const loadTime = perfData.loadEventEnd - perfData.loadEventStart;

                            console.log('Page load time:', loadTime + 'ms');

                            // Track performance metrics
                            if (typeof gtag !== 'undefined' && loadTime > 3000) {
                                gtag('event', 'slow_load', {
                                    'event_category': 'performance',
                                    'event_label': 'page_load_slow',
                                    'value': Math.round(loadTime)
                                });
                            }
                        }
                    }, 0);
                });
            }
        }

        // Initialize performance monitoring
        monitorPerformance();

        // PWA First Load Detection
        function isFirstLoad() {
            return !localStorage.getItem('pwa-visited');
        }

        if (isFirstLoad()) {
            localStorage.setItem('pwa-visited', 'true');
            console.log('First PWA visit detected');

            // Track first visit
            if (typeof gtag !== 'undefined') {
                gtag('event', 'first_pwa_visit', {
                    'event_category': 'engagement',
                    'event_label': 'first_pwa_visit'
                });
            }
        }

        // Cleanup function for PWA features
        window.addEventListener('beforeunload', function() {
            // Clean up any floating elements
            const prompts = document.querySelectorAll('.pwa-install-prompt, .offline-indicator');
            prompts.forEach(prompt => prompt.remove());
        });

        console.log('PWA features initialized successfully');
    </script>
    """

    st.markdown(pwa_registration, unsafe_allow_html=True)

def display_mobile_prompt():
    """Display mobile-specific prompts and optimizations"""
    if st.session_state.get('is_mobile', False):
        st.markdown("""
        <div class="mobile-notice">
            <h4>📱 Mobile Optimize Mode</h4>
            <p>For best results on mobile:</p>
            <ul>
                <li>Use good lighting</li>
                <li>Hold camera steady</li>
                <li>Tap to focus before taking photo</li>
                <li>Include the entire leaf</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def enhanced_display_home_page():
    """Enhanced home page with camera integration and PWA features"""
    # Create two columns for main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📸 Capture or Upload Plant Leaf Image")

        # Tab interface for camera and upload options
        camera_tab, upload_tab = st.tabs(["📷 Camera", "📁 Upload"])

        with camera_tab:
            st.markdown("### 📷 Take Photo with Camera")

            # Camera capture instructions
            st.markdown("""
            <div style="background: #e3f2fd; border-left: 4px solid #2196f3; padding: 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0;">
                <strong>📷 Camera Instructions:</strong><br>
                • Position the leaf clearly in frame<br>
                • Ensure good lighting (avoid shadows)<br>
                • Hold camera steady during capture<br>
                • Include both healthy and affected areas if visible<br>
                • Fill at least 50% of the frame with the leaf
            </div>
            """, unsafe_allow_html=True)

            # Camera capture simulation for demo
            if st.button("📸 Capture Photo", key="capture_photo", type="primary", use_container_width=True):
                with st.spinner("📸 Capturing image..."):
                    time.sleep(2)  # Simulate camera capture
                    st.session_state.camera_captured = True
                    st.success("✅ Photo captured successfully! Analyzing with AI...")
                    st.rerun()

            # Display captured image indicator
            if st.session_state.get('camera_captured', False):
                st.markdown("""
                <div style="background: #d4edda; border: 1px solid #c3e6cb; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                    <strong>✅ Camera Status:</strong> Image captured and ready for analysis<br>
                    <small>💡 Tip: Click "Upload from Captured" below to process the image</small>
                </div>
                """, unsafe_allow_html=True)

            # Simulated file input for camera captured images
            camera_file = st.file_uploader(
                "Upload Captured Photo",
                type=["jpg", "jpeg", "png"],
                help="Upload a photo taken with your camera",
                key="camera_upload"
            )

            if camera_file is not None:
                st.session_state.current_image = camera_file
                st.session_state.camera_captured = False

                # Show loading state while analyzing
                with st.spinner("🔍 Analyzing captured image with AI models..."):
                    try:
                        results = st.session_state.api_client.analyze_image(camera_file)
                        st.session_state.analysis_results = results

                        if results.get('success', False):
                            st.success("✅ Analysis completed successfully!")
                            st.session_state.page_history = ['home', 'results']
                            st.rerun()
                        else:
                            st.error(f"❌ Analysis failed: {results.get('error_message', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.markdown("""
                        **Troubleshooting:**
                        - Check your internet connection
                        - Ensure image is clear and well-lit
                        - Try capturing the image again
                        - Contact support if issue persists
                        """)

        with upload_tab:
            st.markdown("### 📁 Upload Image from Device")

            # Create image uploader component
            uploader = ImageUploader(st.session_state.api_client)

            # Process uploaded image
            if uploader.uploaded_file is not None:
                st.session_state.current_image = uploader.uploaded_file

                # Show loading state while analyzing
                with st.spinner("🔍 Analyzing uploaded image with AI models..."):
                    try:
                        results = st.session_state.api_client.analyze_image(uploader.uploaded_file)
                        st.session_state.analysis_results = results

                        if results.get('success', False):
                            st.success("✅ Analysis completed successfully!")
                            st.session_state.page_history = ['home', 'results']
                            st.rerun()
                        else:
                            st.error(f"❌ Analysis failed: {results.get('error_message', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.markdown("""
                        **Troubleshooting:**
                        - Check your internet connection
                        - Ensure image is clear and well-lit
                        - Try uploading a different image
                        - Contact support if issue persists
                        """)

        # Quick action buttons
        if st.session_state.get('current_image'):
            st.markdown("---")
            col_action1, col_action2, col_action3 = st.columns(3)

            with col_action1:
                if st.button("🔄 Capture New Photo", key="capture_new", use_container_width=True):
                    # Reset for new capture
                    st.session_state.current_image = None
                    st.session_state.camera_captured = False
                    st.session_state.analysis_results = None
                    st.rerun()

            with col_action2:
                if st.button("💾 Save Image", key="save_current", use_container_width=True):
                    # Save current image functionality
                    if st.session_state.get('current_image'):
                        st.download_button(
                            label="Download Image",
                            data=st.session_state.current_image,
                            file_name=f"plant_analysis_{int(time.time())}.jpg",
                            mime="image/jpeg",
                            help="Download captured/uploaded image to your device"
                        )

            with col_action3:
                if st.button("🗑️ Clear", key="clear_current", use_container_width=True):
                    # Clear all image data
                    st.session_state.current_image = None
                    st.session_state.camera_captured = False
                    st.session_state.analysis_results = None
                    st.rerun()

def main():
    """Main application entry point"""
    # Setup page configuration and styling
    setup_page_config()
    setup_custom_css()

    # Register PWA service worker and manifest
    register_pwa_features()

    # Create session state
    create_session_state()

    # Detect mobile (simplified detection)
    user_agent = st.experimental_get_query_params().get('user_agent', '')
    is_mobile = any(mobile in user_agent.lower() for mobile in ['mobile', 'android', 'iphone', 'ipad'])
    st.session_state.is_mobile = is_mobile

    # Display header
    display_header()

    # Display sidebar
    display_sidebar()

    # Main content area based on current page
    current_page = st.session_state.page_history[-1] if st.session_state.page_history else 'home'

    if current_page == 'home':
        display_home_page()
        display_mobile_prompt()
    elif current_page == 'results':
        display_results_page()
    elif current_page == 'treatments':
        display_treatments_page()
    else:
        # Fallback to home page
        display_home_page()
        display_mobile_prompt()

    # Display footer
    display_footer()

    # Add debug information in development mode
    if os.getenv('DEBUG', 'False').lower() == 'true':
        with st.expander("🔧 Debug Information"):
            st.json({
                'session_state': {k: v for k, v in st.session_state.items() if not callable(v)},
                'api_base_url': API_BASE_URL,
                'current_page': current_page,
                'is_mobile': is_mobile,
                'user_agent': user_agent
            })

if __name__ == "__main__":
    main()