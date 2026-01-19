"""
Camera utilities for creating HTML camera components and PWA integration
"""

import base64
import io
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

def create_camera_container(settings: Dict[str, Any]) -> str:
    """Create HTML for camera component with WebRTC integration"""

    default_settings = {
        'width': 640,
        'height': 480,
        'quality': 85,
        'format': 'jpeg',
        'facing_mode': 'environment',
        'auto_capture': False,
        'countdown': False,
        'flash': 'auto',
        'mirror': False,
        'constrain': True,
        'audio': False
    }

    # Merge with provided settings
    camera_settings = {**default_settings, **settings}

    # Create camera HTML
    camera_html = f"""
    <div class="camera-container" id="cameraContainer">
        <!-- Camera Preview Area -->
        <div class="camera-preview">
            <video id="cameraPreview"
                   autoplay
                   playsinline
                   muted
                   width="{camera_settings['width']}"
                   height="{camera_settings['height']}"
                   {f'mirror={"true"}' if camera_settings['mirror'] else ''}>
            </video>

            <!-- Camera Placeholder -->
            <div class="camera-placeholder" id="cameraPlaceholder">
                <div class="camera-icon">📷</div>
                <div class="camera-text">Click "Start Camera" to begin</div>
                <div class="camera-tips">
                    <div class="tip">💡 Ensure good lighting</div>
                    <div class="tip">🌿 Position leaf clearly</div>
                    <div class="tip">📱 Fill the frame</div>
                </div>
            </div>
        </div>

        <!-- Camera Controls -->
        <div class="camera-controls" id="cameraControls" style="display: none;">
            <div class="control-row">
                <button id="captureButton" class="btn btn-primary btn-large">
                    📷 Capture
                </button>
                <button id="stopButton" class="btn btn-secondary btn-large">
                    ⏹️ Stop
                </button>
            </div>

            <div class="control-row">
                <button id="switchCameraButton" class="btn btn-outline">
                    🔄 Switch Camera
                </button>
                <button id="flashButton" class="btn btn-outline">
                    💡 Flash: {camera_settings['flash'].title()}
                </button>
            </div>
        </div>

        <!-- Quality Indicators -->
        <div class="camera-quality" id="cameraQuality" style="display: none;">
            <div class="quality-item">
                <div class="quality-indicator" id="focusIndicator">
                    <span class="indicator-icon">🎯</span>
                    <span class="indicator-text">Focus</span>
                </div>
            </div>
            <div class="quality-item">
                <div class="quality-indicator" id="lightingIndicator">
                    <span class="indicator-icon">☀️</span>
                    <span class="indicator-text">Lighting</span>
                </div>
            </div>
            <div class="quality-item">
                <div class="quality-indicator" id="stabilityIndicator">
                    <span class="indicator-icon">📊</span>
                    <span class="indicator-text">Stability</span>
                </div>
            </div>
        </div>

        <!-- Settings Panel -->
        <div class="camera-settings" id="cameraSettings">
            <div class="setting-group">
                <label for="resolutionSelect">📐 Resolution</label>
                <select id="resolutionSelect" class="form-control">
                    <option value="640x480">640×480 (VGA)</option>
                    <option value="1280x720">1280×720 (HD)</option>
                    <option value="1920x1080">1920×1080 (FHD)</option>
                </select>
            </div>

            <div class="setting-group">
                <label for="qualityRange">📊 Quality</label>
                <input type="range" id="qualityRange" min="50" max="100" value="{camera_settings['quality']}" class="form-control">
                <span id="qualityValue">{camera_settings['quality']}</span>
            </div>

            <div class="setting-group">
                <label for="formatSelect">📁 Format</label>
                <select id="formatSelect" class="form-control">
                    <option value="jpeg">JPEG</option>
                    <option value="png">PNG</option>
                    <option value="webp">WebP</option>
                </select>
            </div>

            <div class="setting-group">
                <label for="whiteBalanceSelect">⚪ White Balance</label>
                <select id="whiteBalanceSelect" class="form-control">
                    <option value="auto">Auto</option>
                    <option value="daylight">Daylight</option>
                    <option value="cloudy">Cloudy</option>
                    <option value="incandescent">Incandescent</option>
                    <option value="fluorescent">Fluorescent</option>
                </select>
            </div>
        </div>

        <!-- Capture Result -->
        <div class="camera-result" id="cameraResult" style="display: none;">
            <img id="capturedImage" alt="Captured plant leaf" />
            <div class="result-info">
                <h3>✅ Image Captured!</h3>
                <div class="result-quality" id="resultQuality">
                    <div class="quality-score" id="qualityScore">--%</div>
                    <div class="quality-label">Quality Score</div>
                </div>
                <div class="result-actions">
                    <button id="useImageButton" class="btn btn-primary">
                        ✅ Use This Image
                    </button>
                    <button id="retakeButton" class="btn btn-secondary">
                        🔄 Retake
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Camera JavaScript -->
    <script>
    // Camera management class
    class CameraManager {{
        constructor(settings) {{
            this.settings = settings;
            this.stream = null;
            this.capturing = false;
            this.facingMode = settings.facing_mode;
            this.track = null;

            this.init();
        }}

        async init() {{
            // Check camera support
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
                this.showError('Camera not supported in this browser');
                return;
            }}

            // Initialize UI
            this.setupEventListeners();
            this.updateSettingsUI();
        }}

        async startCamera() {{
            try {{
                // Stop existing stream
                if (this.stream) {{
                    this.stopCamera();
                }}

                const constraints = this.getConstraints();

                console.log('Starting camera with constraints:', constraints);

                // Show camera placeholder
                this.showPlaceholder(false);
                this.showPreview(true);
                this.showControls(true);

                // Get camera stream
                this.stream = await navigator.mediaDevices.getUserMedia(constraints);

                // Get video track for advanced controls
                this.track = this.stream.getVideoTracks()[0];

                // Apply constraints
                await this.applyTrackConstraints();

                // Start preview
                const video = document.getElementById('cameraPreview');
                video.srcObject = this.stream;

                // Start quality monitoring
                this.startQualityMonitoring();

                console.log('Camera started successfully');

            }} catch (error) {{
                console.error('Camera error:', error);
                this.showError('Failed to start camera: ' + error.message);
            }}
        }}

        async stopCamera() {{
            try {{
                // Stop stream
                if (this.stream) {{
                    this.stream.getTracks().forEach(track => track.stop());
                    this.stream = null;
                }}

                // Clear video
                const video = document.getElementById('cameraPreview');
                video.srcObject = null;

                // Update UI
                this.showPlaceholder(true);
                this.showPreview(false);
                this.showControls(false);
                this.showQualityIndicators(false);

                // Stop quality monitoring
                this.stopQualityMonitoring();

                console.log('Camera stopped');

            }} catch (error) {{
                console.error('Error stopping camera:', error);
            }}
        }}

        getConstraints() {{
            const constraints = {{
                video: {{
                    width: {{ ideal: this.settings.width }},
                    height: {{ ideal: this.settings.height }},
                    facingMode: this.facingMode,
                    deviceId: this.getPreferredDeviceId()
                }},
                audio: this.settings.audio || false
            }};

            // Add advanced constraints if supported
            if (this.settings.constrain) {{
                constraints.video.advanced = [{{
                    width: {{ min: 640 }},
                    height: {{ min: 480 }},
                    aspectRatio: 1.33
                }}];
            }}

            return constraints;
        }}

        async applyTrackConstraints() {{
            if (!this.track) return;

            try {{
                const constraints = {{
                    width: {{ ideal: this.settings.width }},
                    height: {{ ideal: this.settings.height }}
                }};

                // Apply to track if supported
                if ('applyConstraints' in this.track) {{
                    await this.track.applyConstraints(constraints);
                }}

            }} catch (error) {{
                console.warn('Could not apply track constraints:', error);
            }}
        }}

        async captureImage() {{
            if (this.capturing) return;

            this.capturing = true;
            this.updateCaptureButton('Capturing...', true);

            try {{
                const video = document.getElementById('cameraPreview');
                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');

                // Set canvas size
                canvas.width = this.settings.width;
                canvas.height = this.settings.height;

                // Draw video frame
                context.drawImage(video, 0, 0, canvas.width, canvas.height);

                // Apply image quality settings
                const quality = this.settings.quality / 100;
                const format = this.settings.format;

                // Convert to blob
                canvas.toBlob((blob) => {{
                    if (blob) {{
                        this.processCapturedImage(blob);
                    }}
                }}, `image/${{format}}`, quality);

                // Add capture feedback
                this.showCaptureFeedback();

            }} catch (error) {{
                console.error('Capture error:', error);
                this.showError('Failed to capture image: ' + error.message);
            }} finally {{
                this.capturing = false;
                this.updateCaptureButton('📷 Capture', false);
            }}
        }}

        processCapturedImage(blob) {{
            // Convert to base64 for display
            const reader = new FileReader();
            reader.onload = (e) => {{
                const imageData = e.target.result;
                this.displayCapturedImage(imageData, blob.size);

                // Assess image quality
                this.assessImageQuality(imageData);

                // Streamlit integration
                this.sendToStreamlit(imageData);
            }};
            reader.readAsDataURL(blob);
        }}

        displayCapturedImage(imageData, fileSize) {{
            // Show captured image
            const image = document.getElementById('capturedImage');
            image.src = imageData;

            // Update result section
            this.showResult(true);
            this.showPreview(false);
            this.showControls(false);

            // Update file size info
            const fileSizeMB = (fileSize / (1024 * 1024)).toFixed(1);
            document.getElementById('resultInfo').innerHTML = `
                <div class="file-info">File Size: ${{fileSizeMB}} MB</div>
                <div class="file-info">Format: ${{this.settings.format.toUpperCase()}}</div>
                <div class="file-info">Resolution: ${{this.settings.width}}×${{this.settings.height}}</div>
            `;
        }}

        assessImageQuality(imageData) {{
            // Simple quality assessment
            const img = new Image();
            img.onload = () => {{
                const quality = this.calculateImageQuality(img);
                this.updateQualityDisplay(quality);
            }};
            img.src = imageData;
        }}

        calculateImageQuality(img) {{
            // Basic quality assessment
            let score = 80; // Start with good score

            // Size factor
            if (img.width < 512 || img.height < 512) {{
                score -= 30;
            }}

            // Aspect ratio check
            const aspectRatio = img.width / img.height;
            if (aspectRatio < 0.5 || aspectRatio > 2.0) {{
                score -= 10;
            }}

            return Math.max(0, Math.min(100, score));
        }}

        updateQualityDisplay(score) {{
            const scoreElement = document.getElementById('qualityScore');
            scoreElement.textContent = score + '%';

            // Update color based on score
            if (score >= 80) {{
                scoreElement.style.color = '#4CAF50';
            }} else if (score >= 60) {{
                scoreElement.style.color = '#FF9800';
            }} else {{
                scoreElement.style.color = '#F44336';
            }}
        }}

        showCaptureFeedback() {{
            // Flash effect
            const video = document.getElementById('cameraPreview');
            video.style.opacity = '0.3';
            setTimeout(() => {{
                video.style.opacity = '1';
            }}, 100);
        }}

        async switchCamera() {{
            // Toggle facing mode
            this.facingMode = this.facingMode === 'environment' ? 'user' : 'environment';

            // Restart camera with new facing mode
            if (this.stream) {{
                await this.stopCamera();
                await this.startCamera();
            }}

            // Update button text
            const switchBtn = document.getElementById('switchCameraButton');
            switchBtn.textContent = this.facingMode === 'environment' ? '📱 Front' : '🔄 Back';
        }}

        startQualityMonitoring() {{
            this.qualityInterval = setInterval(() => {{
                if (this.stream && this.track) {{
                    this.updateQualityIndicators();
                }}
            }}, 1000);
        }}

        stopQualityMonitoring() {{
            if (this.qualityInterval) {{
                clearInterval(this.qualityInterval);
                this.qualityInterval = null;
            }}
        }}

        updateQualityIndicators() {{
            // Get track settings
            const settings = this.track.getSettings();

            // Update indicators based on track state
            this.updateIndicator('focusIndicator', settings.focusMode !== 'none' ? '🟢' : '🔴');
            this.updateIndicator('lightingIndicator', settings.brightness > 0.5 ? '🟢' : '🔴');
            this.updateIndicator('stabilityIndicator', this.stream.active ? '🟢' : '🔴');
        }}

        updateIndicator(indicatorId, status) {{
            const indicator = document.getElementById(indicatorId);
            if (indicator) {{
                const icon = indicator.querySelector('.indicator-icon');
                const text = indicator.querySelector('.indicator-text');

                if (status === '🟢') {{
                    icon.textContent = icon.textContent.replace(/[🔴🟡]/g, '🟢');
                    text.style.color = '#4CAF50';
                }} else if (status === '🔴') {{
                    icon.textContent = icon.textContent.replace(/[🟢🟡]/g, '🔴');
                    text.style.color = '#F44336';
                }}
            }}
        }}

        showPlaceholder(show) {{
            const placeholder = document.getElementById('cameraPlaceholder');
            const preview = document.getElementById('cameraPreview');

            placeholder.style.display = show ? 'flex' : 'none';
            preview.style.display = show ? 'none' : 'block';
        }}

        showPreview(show) {{
            const preview = document.getElementById('cameraPreview');
            const controls = document.getElementById('cameraControls');

            preview.style.display = show ? 'block' : 'none';
            controls.style.display = show ? 'block' : 'none';
        }}

        showControls(show) {{
            const controls = document.getElementById('cameraControls');
            controls.style.display = show ? 'block' : 'none';
        }}

        showQualityIndicators(show) {{
            const quality = document.getElementById('cameraQuality');
            quality.style.display = show ? 'flex' : 'none';
        }}

        showResult(show) {{
            const result = document.getElementById('cameraResult');
            result.style.display = show ? 'block' : 'none';
        }}

        updateCaptureButton(text, disabled) {{
            const button = document.getElementById('captureButton');
            button.textContent = text;
            button.disabled = disabled;
        }}

        updateSettingsUI() {{
            // Update settings controls with current values
            document.getElementById('qualityRange').value = this.settings.quality;
            document.getElementById('qualityValue').textContent = this.settings.quality;
            document.getElementById('formatSelect').value = this.settings.format;
        }}

        getPreferredDeviceId() {{
            // Check for saved device preference
            const savedDevice = localStorage.getItem('preferredCameraDevice');
            return savedDevice || undefined;
        }}

        saveDevicePreference(deviceId) {{
            localStorage.setItem('preferredCameraDevice', deviceId);
        }}

        sendToStreamlit(imageData) {{
            // Send captured image to Streamlit
            if (window.parent && window.parent.Streamlit) {{
                window.parent.Streamlit.setComponentValue({{
                    image_data: imageData,
                    camera_settings: this.settings,
                    timestamp: Date.now()
                }});
            }}
        }}

        showError(message) {{
            // Show error message
            const errorDiv = document.createElement('div');
            errorDiv.className = 'camera-error';
            errorDiv.innerHTML = `
                <div class="error-icon">⚠️</div>
                <div class="error-message">${{message}}</div>
                <button class="btn btn-primary" onclick="cameraManager.startCamera()">Retry</button>
            `;

            document.getElementById('cameraContainer').appendChild(errorDiv);

            // Remove error after 5 seconds
            setTimeout(() => {{
                if (errorDiv.parentNode) {{
                    errorDiv.parentNode.removeChild(errorDiv);
                }}
            }}, 5000);
        }}

        setupEventListeners() {{
            // Capture button
            document.getElementById('captureButton').addEventListener('click', () => {{
                this.captureImage();
            }});

            // Stop button
            document.getElementById('stopButton').addEventListener('click', () => {{
                this.stopCamera();
            }});

            // Switch camera button
            document.getElementById('switchCameraButton').addEventListener('click', () => {{
                this.switchCamera();
            }});

            // Settings controls
            document.getElementById('qualityRange').addEventListener('input', (e) => {{
                this.settings.quality = parseInt(e.target.value);
                document.getElementById('qualityValue').textContent = e.target.value;
            }});

            document.getElementById('formatSelect').addEventListener('change', (e) => {{
                this.settings.format = e.target.value;
            }});

            // Result buttons
            document.getElementById('useImageButton').addEventListener('click', () => {{
                this.sendToStreamlit(document.getElementById('capturedImage').src);
            }});

            document.getElementById('retakeButton').addEventListener('click', () => {{
                this.hideResult();
                this.startCamera();
            }});
        }}

        hideResult() {{
            this.showResult(false);
            this.showPreview(true);
            this.showControls(true);
        }}
    }}

    // Initialize camera when DOM is ready
    document.addEventListener('DOMContentLoaded', () => {{
        const settings = {json.dumps(camera_settings)};
        window.cameraManager = new CameraManager(settings);
    }});

    // Handle camera permissions
    async function checkCameraPermissions() {{
        try {{
            const permission = await navigator.permissions.query({{ name: 'camera' }});
            console.log('Camera permission state:', permission.state);

            permission.addEventListener('change', () => {{
                console.log('Camera permission changed:', permission.state);
            }});

        }} catch (error) {{
            console.log('Permission API not supported:', error);
        }}
    }}

    // Auto-start camera if requested
    if ({str(camera_settings.get('auto_capture', False)).lower() == 'true') {{
        document.addEventListener('DOMContentLoaded', () => {{
            setTimeout(() => {{
                if (window.cameraManager) {{
                    window.cameraManager.startCamera();
                }}
            }}, 1000);
        }});
    }}
    </script>
    """

    return camera_html


def create_pwa_camera_wrapper() -> str:
    """Create PWA-specific camera wrapper with offline support"""

    pwa_html = """
    <div class="pwa-camera-wrapper">
        <!-- PWA Status -->
        <div class="pwa-status" id="pwaStatus">
            <div class="status-indicator" id="statusIndicator">
                <span class="indicator-online" id="onlineIndicator">🟢</span>
                <span class="status-text" id="statusText">Ready</span>
            </div>
        </div>

        <!-- Offline Banner -->
        <div class="offline-banner" id="offlineBanner" style="display: none;">
            <span class="offline-icon">⚠️</span>
            <span class="offline-text">You're offline - using cached camera features</span>
        </div>

        <!-- Install Prompt -->
        <div class="install-prompt" id="installPrompt" style="display: none;">
            <div class="prompt-content">
                <span class="prompt-icon">📱</span>
                <span class="prompt-text">Install Plant Disease Detector</span>
                <button class="btn btn-primary" id="installButton">Install</button>
                <button class="btn btn-outline" id="dismissButton">Maybe Later</button>
            </div>
        </div>

        <!-- Camera Component Container -->
        <div id="cameraComponentContainer">
            <!-- Camera component will be inserted here -->
        </div>
    </div>

    <script>
    // PWA Camera Integration
    class PWACameraManager {{
        constructor() {{
            this.isOnline = navigator.onLine;
            this.isInstalled = window.matchMedia('(display-mode: standalone)').matches;

            this.init();
        }}

        init() {{
            // Monitor online status
            window.addEventListener('online', () => this.updateOnlineStatus(true));
            window.addEventListener('offline', () => this.updateOnlineStatus(false));

            // Update initial status
            this.updateOnlineStatus(this.isOnline);

            // Check for PWA installation
            this.checkInstallPrompt();

            // Setup background sync
            this.setupBackgroundSync();
        }}

        updateOnlineStatus(isOnline) {{
            this.isOnline = isOnline;
            const offlineBanner = document.getElementById('offlineBanner');
            const onlineIndicator = document.getElementById('onlineIndicator');
            const statusText = document.getElementById('statusText');

            if (isOnline) {{
                offlineBanner.style.display = 'none';
                onlineIndicator.textContent = '🟢';
                statusText.textContent = 'Online';
                statusText.style.color = '#4CAF50';
            }} else {{
                offlineBanner.style.display = 'flex';
                onlineIndicator.textContent = '🔴';
                statusText.textContent = 'Offline';
                statusText.style.color = '#F44336';
            }}
        }}

        checkInstallPrompt() {{
            // Check if app is already installed
            if (this.isInstalled) {{
                return;
            }}

            // Show install prompt if not installed and browser supports it
            if ('beforeinstallprompt' in window) {{
                window.addEventListener('beforeinstallprompt', (e) => {{
                    e.preventDefault();
                    this.showInstallPrompt(e);
                }});
            }}
        }}

        showInstallPrompt(event) {{
            const prompt = document.getElementById('installPrompt');
            prompt.style.display = 'flex';

            const installBtn = document.getElementById('installButton');
            const dismissBtn = document.getElementById('dismissButton');

            installBtn.addEventListener('click', async () => {{
                const result = await event.prompt();
                if (result.outcome === 'accepted') {{
                    this.hideInstallPrompt();
                }}
            }});

            dismissBtn.addEventListener('click', () => {{
                this.hideInstallPrompt();
            }});

            // Auto-hide after 10 seconds
            setTimeout(() => {{
                this.hideInstallPrompt();
            }}, 10000);
        }}

        hideInstallPrompt() {{
            document.getElementById('installPrompt').style.display = 'none';
        }}

        setupBackgroundSync() {{
            if ('serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype) {{
                navigator.serviceWorker.ready.then(registration => {{
                    // Register for background sync
                    this.backgroundSyncRegistration = registration;
                }}).catch(error => {{
                    console.log('Background sync not available:', error);
                }});
            }}
        }}

        async registerBackgroundSync(tag) {{
            if (this.backgroundSyncRegistration && 'sync' in this.backgroundSyncRegistration) {{
                try {{
                    await this.backgroundSyncRegistration.sync.register(tag);
                    console.log('Background sync registered for:', tag);
                }} catch (error) {{
                    console.log('Background sync registration failed:', error);
                }}
            }}
        }}

        getOfflineCameraSettings() {{
            // Get cached camera settings for offline use
            return {
                width: 640,
                height: 480,
                quality: 75,
                format: 'jpeg',
                facing_mode: 'environment'
            };
        }}

        cacheCameraCapture(imageData) {{
            // Cache captured image for offline access
            if ('caches' in window) {{
                const cacheName = 'camera-captures';
                caches.open(cacheName).then(cache => {{
                    const response = new Response(imageData, {{
                        headers: {{ 'Content-Type': 'image/jpeg' }}
                    }});
                    cache.put('/last-capture', response);
                    console.log('Camera capture cached for offline use');
                }});
            }}
        }}

        async getCachedCameraCapture() {{
            // Get last cached camera capture
            if ('caches' in window) {{
                try {{
                    const cache = await caches.open('camera-captures');
                    const response = await cache.match('/last-capture');
                    if (response) {{
                        return await response.text();
                    }}
                }} catch (error) {{
                    console.log('No cached camera capture available');
                }}
            }}
            return null;
        }}
    }}

    // Initialize PWA camera manager
    document.addEventListener('DOMContentLoaded', () => {{
        window.pwaCameraManager = new PWACameraManager();
    }});
    </script>
    """

    return pwa_html


def get_camera_constraints() -> Dict[str, Any]:
    """Get WebRTC camera constraints based on device capabilities"""

    constraints = {
        'video': {
            'width': {'ideal': 1280, 'max': 1920},
            'height': {'ideal': 720, 'max': 1080},
            'facingMode': 'environment',
            'aspectRatio': 1.7777777778  // 16:9
        },
        'audio': False
    }

    # Add advanced constraints if supported
    constraints['video']['advanced'] = [
        {
            'width': {'min': 640},
            'height': {'min': 480},
            'aspectRatio': 1.3333333333,  // 4:3 fallback
            'facingMode': 'environment'
        }
    ]

    return constraints


def optimize_for_pwa() -> Dict[str, Any]:
    """Get PWA-optimized camera settings"""

    return {
        'width': 640,
        'height': 480,
        'quality': 75,
        'format': 'jpeg',
        'facing_mode': 'environment',
        'auto_capture': False,
        'countdown': False,
        'flash': 'auto',
        'mirror': False,
        'constrain': True,
        'audio': False,
        'pwa_mode': True,
        'cache_enabled': True,
        'background_sync': True
    }


def check_mobile_camera_support() -> bool:
    """Check if mobile camera APIs are available"""

    # Basic checks
    if not hasattr(window, 'navigator') or not hasattr(navigator, 'mediaDevices'):
        return False

    # Check for getUserMedia support
    if not hasattr(navigator.mediaDevices, 'getUserMedia'):
        return False

    # Check for mobile-specific features
    user_agent = navigator.userAgent.lower()
    is_mobile = any(mobile in user_agent for mobile in ['mobile', 'android', 'iphone'])

    return is_mobile


def create_mobile_camera_overlay() -> str:
    """Create mobile-specific camera overlay UI"""

    mobile_overlay = """
    <div class="mobile-camera-overlay" id="mobileCameraOverlay">
        <!-- Mobile Specific Controls -->
        <div class="mobile-controls">
            <!-- Touch to capture area -->
            <div class="touch-capture-area" id="touchCaptureArea">
                <div class="capture-bracket">
                    <div class="bracket-corner top-left"></div>
                    <div class="bracket-corner top-right"></div>
                    <div class="bracket-corner bottom-left"></div>
                    <div class="bracket-corner bottom-right"></div>
                </div>
                <div class="capture-hint">Tap to capture</div>
            </div>

            <!-- Quick settings -->
            <div class="quick-settings">
                <button class="mobile-btn mobile-btn-primary" id="mobileCaptureBtn">
                    📷 Capture
                </button>
                <button class="mobile-btn mobile-btn-secondary" id="mobileSwitchBtn">
                    🔄 Camera
                </button>
                <button class="mobile-btn mobile-btn-outline" id="mobileFlashBtn">
                    💡 Flash
                </button>
            </div>
        </div>

        <!-- Mobile quality indicators -->
        <div class="mobile-quality">
            <div class="mobile-quality-item">
                <span class="mobile-indicator" id="mobileFocusIndicator">🎯</span>
                <span class="mobile-label">Focus</span>
            </div>
            <div class="mobile-quality-item">
                <span class="mobile-indicator" id="mobileLightingIndicator">☀️</span>
                <span class="mobile-label">Light</span>
            </div>
            <div class="mobile-quality-item">
                <span class="mobile-indicator" id="mobileStabilityIndicator">📊</span>
                <span class="mobile-label">Stable</span>
            </div>
        </div>
    </div>

    <script>
    // Mobile Camera Touch Support
    class MobileCameraHandler {{
        constructor() {{
            this.touchStartX = 0;
            this.touchStartY = 0;
            this.touchEndX = 0;
            this.touchEndY = 0;

            this.init();
        }}

        init() {{
            const touchArea = document.getElementById('touchCaptureArea');

            // Touch events
            touchArea.addEventListener('touchstart', (e) => this.handleTouchStart(e), {{ passive: true }});
            touchArea.addEventListener('touchend', (e) => this.handleTouchEnd(e), {{ passive: true }});

            // Click fallback
            touchArea.addEventListener('click', () => this.handleCapture());
        }}

        handleTouchStart(e) {{
            this.touchStartX = e.touches[0].clientX;
            this.touchStartY = e.touches[0].clientY;

            // Add visual feedback
            e.target.classList.add('touch-active');
        }}

        handleTouchEnd(e) {{
            this.touchEndX = e.changedTouches[0].clientX;
            this.touchEndY = e.changedTouches[0].clientY;

            // Remove visual feedback
            e.target.classList.remove('touch-active');

            // Check if it's a tap (not swipe)
            const deltaX = Math.abs(this.touchEndX - this.touchStartX);
            const deltaY = Math.abs(this.touchEndY - this.touchStartY);

            if (deltaX < 10 && deltaY < 10) {{
                this.handleCapture();
            }}
        }}

        handleCapture() {{
            // Add haptic feedback if supported
            if ('vibrate' in navigator) {{
                navigator.vibrate(50); // Short vibration
            }}

            // Trigger capture
            if (window.cameraManager) {{
                window.cameraManager.captureImage();
            }}

            // Visual feedback
            this.showCaptureFeedback();
        }}

        showCaptureFeedback() {{
            const touchArea = document.getElementById('touchCaptureArea');

            // Flash effect
            touchArea.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
            touchArea.style.transform = 'scale(0.95)';

            setTimeout(() => {{
                touchArea.style.backgroundColor = 'transparent';
                touchArea.style.transform = 'scale(1)';
            }}, 200);
        }}

        // Setup mobile button handlers
        setupMobileButtons() {{
            document.getElementById('mobileCaptureBtn').addEventListener('click', () => {{
                this.handleCapture();
            }});

            document.getElementById('mobileSwitchBtn').addEventListener('click', () => {{
                if (window.cameraManager) {{
                    window.cameraManager.switchCamera();
                }}
            }});

            document.getElementById('mobileFlashBtn').addEventListener('click', () => {{
                // Toggle flash (implementation dependent on device)
                this.toggleFlash();
            }});
        }}

        toggleFlash() {{
            // Flash toggle implementation would be device-specific
            console.log('Flash toggle requested');
        }}
    }}

    // Initialize mobile camera when DOM is ready
    document.addEventListener('DOMContentLoaded', () => {{
        window.mobileCameraHandler = new MobileCameraHandler();
        window.mobileCameraHandler.setupMobileButtons();
    }});
    </script>
    """

    return mobile_overlay