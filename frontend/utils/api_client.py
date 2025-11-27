"""
API Client Utility

This module handles communication between the Streamlit frontend and FastAPI backend,
including image upload, analysis requests, and data retrieval.
"""

import asyncio
import io
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

import httpx
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

from frontend.utils.image_processing import resize_for_display, create_thumbnail

logger = logging.getLogger(__name__)


class APIClient:
    """Client for communicating with Plant Disease Detector backend API"""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """Initialize API client with backend URL and timeout"""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

        # Configure session
        self.session.headers.update({
            'User-Agent': 'PlantDiseaseDetector-Frontend/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

        logger.info(f"API client initialized with base URL: {self.base_url}")

    def _build_url(self, endpoint: str) -> str:
        """Build complete URL for API endpoint"""
        return f"{self.base_url}{endpoint}"

    def _handle_response(self, response: requests.Response, endpoint: str) -> Dict[str, Any]:
        """Handle API response with error handling"""
        try:
            response.raise_for_status()

            # Try to parse JSON
            try:
                data = response.json()
            except json.JSONDecodeError:
                logger.warning(f"Non-JSON response from {endpoint}: {response.text[:200]}...")
                return {
                    'success': False,
                    'error': 'Invalid JSON response from server',
                    'status_code': response.status_code,
                    'raw_response': response.text[:500]
                }

            # Log successful response
            logger.info(f"API response from {endpoint}: {response.status_code}")

            return data

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error: {str(e)}"
            logger.error(f"API error from {endpoint}: {error_msg}")

            return {
                'success': False,
                'error': error_msg,
                'status_code': response.status_code if hasattr(response, 'status_code') else None,
                'url': self._build_url(endpoint)
            }

        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            logger.error(f"API request failed for {endpoint}: {error_msg}")

            return {
                'success': False,
                'error': error_msg,
                'url': self._build_url(endpoint)
            }

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error from {endpoint}: {error_msg}")

            return {
                'success': False,
                'error': error_msg,
                'url': self._build_url(endpoint)
            }

    async def analyze_image(self, image_file: Union[bytes, io.BytesIO]) -> Dict[str, Any]:
        """Upload and analyze plant leaf image"""
        try:
            # Prepare file for upload
            if isinstance(image_file, bytes):
                files = {'image': ('leaf.jpg', io.BytesIO(image_file), 'image/jpeg')}
            else:
                image_file.seek(0)  # Reset file pointer
                files = {'image': image_file}

            # Add optional parameters
            data = {
                'quality_threshold': 0.3,
                'include_alternatives': True,
                'max_processing_time_ms': 30000
            }

            logger.info(f"Sending image for analysis, size: {len(image_file) if isinstance(image_file, bytes) else 'unknown'} bytes")

            # Make request
            response = self.session.post(
                self._build_url("/api/v1/analyze"),
                files=files,
                data=data,
                timeout=self.timeout
            )

            return self._handle_response(response, "/api/v1/analyze")

        except Exception as e:
            logger.error(f"Failed to analyze image: {str(e)}")
            return {
                'success': False,
                'error': f'Analysis failed: {str(e)}',
                'processing_time_ms': 0
            }

    async def batch_analyze_images(self, image_files: List[Union[bytes, io.BytesIO]]) -> Dict[str, Any]:
        """Analyze multiple images in batch"""
        try:
            # Prepare files for upload
            files = []
            for i, image_file in enumerate(image_files):
                if isinstance(image_file, bytes):
                    files.append(('images', f'image_{i}.jpg', io.BytesIO(image_file), 'image/jpeg'))
                else:
                    image_file.seek(0)
                    files.append(('images', image_file))

            # Add batch parameters
            data = {
                'max_concurrent': 3,
                'timeout_per_image_ms': 30000,
                'fail_fast': True
            }

            logger.info(f"Sending batch of {len(image_files)} images for analysis")

            # Make request
            response = self.session.post(
                self._build_url("/api/v1/analyze/batch"),
                files=files,
                data=data,
                timeout=self.timeout * 2  # Longer timeout for batch
            )

            return self._handle_response(response, "/api/v1/analyze/batch")

        except Exception as e:
            logger.error(f"Failed to batch analyze images: {str(e)}")
            return {
                'success': False,
                'error': f'Batch analysis failed: {str(e)}',
                'total_images': len(image_files),
                'successful_analyses': 0,
                'failed_analyses': len(image_files)
            }

    def get_disease_info(self, disease_id: str, include_treatments: bool = True, include_images: bool = True) -> Dict[str, Any]:
        """Get detailed information about a specific disease"""
        try:
            params = {
                'include_treatments': str(include_treatments).lower(),
                'include_images': str(include_images).lower()
            }

            logger.info(f"Fetching disease info for: {disease_id}")

            response = self.session.get(
                self._build_url(f"/api/v1/diseases/{disease_id}"),
                params=params,
                timeout=self.timeout
            )

            return self._handle_response(response, f"/api/v1/diseases/{disease_id}")

        except Exception as e:
            logger.error(f"Failed to get disease info for {disease_id}: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to retrieve disease information: {str(e)}',
                'disease_id': disease_id
            }

    def get_reference_image_url(self, disease_id: str, image_index: int = 0) -> str:
        """Generate URL for disease reference image"""
        return f"{self._build_url('/api/v1/images/reference')}/{disease_id}?image_index={image_index}"

    def get_reference_image(self, disease_id: str, image_index: int = 0) -> Optional[bytes]:
        """Download reference image for a disease"""
        try:
            url = self.get_reference_image_url(disease_id, image_index)

            logger.info(f"Fetching reference image: {disease_id}, index: {image_index}")

            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            return response.content

        except Exception as e:
            logger.error(f"Failed to get reference image {disease_id}[{image_index}]: {str(e)}")
            return None

    def search_diseases(self, query: str, limit: int = 10, plant: Optional[str] = None) -> Dict[str, Any]:
        """Search diseases by name, symptoms, or causes"""
        try:
            params = {'limit': limit}
            if query:
                params['q'] = query
            if plant:
                params['plant'] = plant

            logger.info(f"Searching diseases with query: '{query}', limit: {limit}")

            response = self.session.get(
                self._build_url("/api/v1/search/diseases"),
                params=params,
                timeout=self.timeout
            )

            return self._handle_response(response, "/api/v1/search/diseases")

        except Exception as e:
            logger.error(f"Failed to search diseases: {str(e)}")
            return {
                'success': False,
                'error': f'Search failed: {str(e)}',
                'query': query,
                'limit': limit
            }

    def get_supported_plants(self) -> Dict[str, Any]:
        """Get list of all supported plants"""
        try:
            logger.info("Fetching supported plants list")

            response = self.session.get(
                self._build_url("/api/v1/plants/supported"),
                timeout=self.timeout
            )

            return self._handle_response(response, "/api/v1/plants/supported")

        except Exception as e:
            logger.error(f"Failed to get supported plants: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to retrieve supported plants: {str(e)}'
            }

    def get_disease_categories(self) -> Dict[str, Any]:
        """Get diseases categorized by crop type"""
        try:
            logger.info("Fetching disease categories")

            response = self.session.get(
                self._build_url("/api/v1/diseases/categories"),
                timeout=self.timeout
            )

            return self._handle_response(response, "/api/v1/diseases/categories")

        except Exception as e:
            logger.error(f"Failed to get disease categories: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to retrieve disease categories: {str(e)}'
            }

    def get_models_status(self) -> Dict[str, Any]:
        """Get status of all AI models"""
        try:
            logger.info("Fetching AI models status")

            response = self.session.get(
                self._build_url("/api/v1/models/status"),
                timeout=self.timeout
            )

            return self._handle_response(response, "/api/v1/models/status")

        except Exception as e:
            logger.error(f"Failed to get models status: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to retrieve models status: {str(e)}'
            }

    def get_api_statistics(self) -> Dict[str, Any]:
        """Get comprehensive API statistics"""
        try:
            logger.info("Fetching API statistics")

            response = self.session.get(
                self._build_url("/api/v1/stats"),
                timeout=self.timeout
            )

            return self._handle_response(response, "/api/v1/stats")

        except Exception as e:
            logger.error(f"Failed to get API statistics: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to retrieve API statistics: {str(e)}'
            }

    def export_diseases(self) -> Dict[str, Any]:
        """Export complete disease database"""
        try:
            logger.info("Exporting disease database")

            response = self.session.get(
                self._build_url("/api/v1/export/diseases"),
                timeout=self.timeout * 2  # Longer timeout for export
            )

            return self._handle_response(response, "/api/v1/export/diseases")

        except Exception as e:
            logger.error(f"Failed to export diseases: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to export disease database: {str(e)}'
            }

    def get_health(self) -> Dict[str, Any]:
        """Check API health and service status"""
        try:
            logger.info("Checking API health")

            response = self.session.get(
                self._build_url("/api/v1/health"),
                timeout=10  # Shorter timeout for health check
            )

            return self._handle_response(response, "/api/v1/health")

        except Exception as e:
            logger.error(f"Failed to check API health: {str(e)}")
            return {
                'success': False,
                'error': f'Health check failed: {str(e)}',
                'services': {}
            }

    def check_connectivity(self) -> Tuple[bool, str]:
        """Check basic connectivity to backend"""
        try:
            # Try a simple health check
            response = self.session.get(
                self._build_url("/api/v1/health"),
                timeout=5  # Very short timeout
            )

            if response.status_code == 200:
                logger.info("Backend connectivity check: SUCCESS")
                return True, "Connected successfully"
            else:
                logger.warning(f"Backend connectivity check: HTTP {response.status_code}")
                return False, f"Server returned HTTP {response.status_code}"

        except requests.exceptions.ConnectionError:
            logger.error("Backend connectivity check: CONNECTION ERROR")
            return False, "Cannot connect to backend server"
        except requests.exceptions.Timeout:
            logger.error("Backend connectivity check: TIMEOUT")
            return False, "Connection to backend timed out"
        except Exception as e:
            logger.error(f"Backend connectivity check: {str(e)}")
            return False, f"Connection error: {str(e)}"

    def set_timeout(self, timeout: int):
        """Update request timeout"""
        self.timeout = timeout
        logger.info(f"API client timeout updated to: {timeout} seconds")

    def set_base_url(self, base_url: str):
        """Update backend base URL"""
        self.base_url = base_url.rstrip('/')
        logger.info(f"API client base URL updated to: {self.base_url}")

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        return {
            'base_url': self.base_url,
            'timeout': self.timeout,
            'headers': dict(self.session.headers)
        }

    def close(self):
        """Close the session and clean up resources"""
        if self.session:
            self.session.close()
            logger.info("API client session closed")


class AsyncAPIClient:
    """Async version of API client for concurrent operations"""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """Initialize async API client"""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

        logger.info(f"Async API client initialized with base URL: {self.base_url}")

    def _build_url(self, endpoint: str) -> str:
        """Build complete URL for API endpoint"""
        return f"{self.base_url}{endpoint}"

    async def _handle_response(self, response: httpx.Response, endpoint: str) -> Dict[str, Any]:
        """Handle async API response with error handling"""
        try:
            response.raise_for_status()

            # Try to parse JSON
            try:
                data = response.json()
            except json.JSONDecodeError:
                logger.warning(f"Non-JSON response from {endpoint}: {response.text[:200]}...")
                return {
                    'success': False,
                    'error': 'Invalid JSON response from server',
                    'status_code': response.status_code,
                    'raw_response': response.text[:500]
                }

            # Log successful response
            logger.info(f"Async API response from {endpoint}: {response.status_code}")

            return data

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error: {str(e)}"
            logger.error(f"Async API error from {endpoint}: {error_msg}")

            return {
                'success': False,
                'error': error_msg,
                'status_code': response.status_code,
                'url': self._build_url(endpoint)
            }

        except httpx.RequestError as e:
            error_msg = f"Request error: {str(e)}"
            logger.error(f"Async API request failed for {endpoint}: {error_msg}")

            return {
                'success': False,
                'error': error_msg,
                'url': self._build_url(endpoint)
            }

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected async error from {endpoint}: {error_msg}")

            return {
                'success': False,
                'error': error_msg,
                'url': self._build_url(endpoint)
            }

    async def analyze_image_async(self, image_file: Union[bytes, io.BytesIO]) -> Dict[str, Any]:
        """Async version of image analysis"""
        try:
            # Prepare file for upload
            if isinstance(image_file, bytes):
                files = {'image': ('leaf.jpg', io.BytesIO(image_file), 'image/jpeg')}
            else:
                image_file.seek(0)
                files = {'image': image_file}

            # Add optional parameters
            data = {
                'quality_threshold': 0.3,
                'include_alternatives': True,
                'max_processing_time_ms': 30000
            }

            logger.info(f"Sending image for async analysis, size: {len(image_file) if isinstance(image_file, bytes) else 'unknown'} bytes")

            # Make async request
            response = await self.client.post(
                self._build_url("/api/v1/analyze"),
                files=files,
                data=data
            )

            return await self._handle_response(response, "/api/v1/analyze")

        except Exception as e:
            logger.error(f"Failed to analyze image async: {str(e)}")
            return {
                'success': False,
                'error': f'Async analysis failed: {str(e)}',
                'processing_time_ms': 0
            }

    async def batch_analyze_images_async(self, image_files: List[Union[bytes, io.BytesIO]]) -> Dict[str, Any]:
        """Async version of batch image analysis"""
        try:
            # Prepare files for upload
            files = []
            for i, image_file in enumerate(image_files):
                if isinstance(image_file, bytes):
                    files.append(('images', f'image_{i}.jpg', io.BytesIO(image_file), 'image/jpeg'))
                else:
                    image_file.seek(0)
                    files.append(('images', image_file))

            # Add batch parameters
            data = {
                'max_concurrent': 3,
                'timeout_per_image_ms': 30000,
                'fail_fast': True
            }

            logger.info(f"Sending batch of {len(image_files)} images for async analysis")

            # Make async request
            response = await self.client.post(
                self._build_url("/api/v1/analyze/batch"),
                files=files,
                data=data
            )

            return await self._handle_response(response, "/api/v1/analyze/batch")

        except Exception as e:
            logger.error(f"Failed to batch analyze images async: {str(e)}")
            return {
                'success': False,
                'error': f'Async batch analysis failed: {str(e)}',
                'total_images': len(image_files),
                'successful_analyses': 0,
                'failed_analyses': len(image_files)
            }

    async def close(self):
        """Close the async client and clean up resources"""
        await self.client.aclose()
        logger.info("Async API client closed")


class CacheManager:
    """Simple cache manager for API responses"""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """Initialize cache with size and TTL"""
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_times = {}

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.access_times:
            return True

        age = time.time() - self.access_times[key]
        return age > self.ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if key in self.cache and not self._is_expired(key):
            self.access_times[key] = time.time()
            logger.debug(f"Cache hit for key: {key}")
            return self.cache[key]

        logger.debug(f"Cache miss for key: {key}")
        return None

    def set(self, key: str, value: Any) -> None:
        """Set cached value"""
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[key] = value
        self.access_times[key] = time.time()
        logger.debug(f"Cache set for key: {key}")

    def clear(self) -> None:
        """Clear all cached entries"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'entries': list(self.cache.keys())
        }


# Utility functions for common operations
def create_api_client(base_url: str = None, use_async: bool = False) -> Union[APIClient, AsyncAPIClient]:
    """Create API client with appropriate configuration"""
    base_url = base_url or os.getenv("BACKEND_URL", "http://localhost:8000")
    timeout = int(os.getenv("API_TIMEOUT_SECONDS", "30"))

    if use_async:
        return AsyncAPIClient(base_url, timeout)
    else:
        return APIClient(base_url, timeout)


def validate_api_response(response: Dict[str, Any], required_fields: List[str] = None) -> Tuple[bool, Optional[str]]:
    """Validate API response structure"""
    if not isinstance(response, dict):
        return False, "Response is not a dictionary"

    if 'success' not in response:
        return False, "Response missing 'success' field"

    if not response['success']:
        error_msg = response.get('error', 'Unknown error')
        return False, f"API returned error: {error_msg}"

    if required_fields:
        for field in required_fields:
            if field not in response:
                return False, f"Response missing required field: {field}"

    return True, None


def format_analysis_time(processing_time_ms: int) -> str:
    """Format processing time for display"""
    if processing_time_ms < 1000:
        return f"{processing_time_ms}ms"
    elif processing_time_ms < 60000:
        return f"{processing_time_ms / 1000:.1f}s"
    else:
        minutes = processing_time_ms / 60000
        return f"{minutes:.1f}min"


def calculate_confidence_color(confidence: float) -> str:
    """Get color code based on confidence level"""
    if confidence >= 0.8:
        return "#4CAF50"  # Green
    elif confidence >= 0.6:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red


def create_error_display(error_message: str, retry_action: Optional[str] = None) -> str:
    """Create formatted error display"""
    error_html = f"""
    <div class="error-container">
        <div class="error-icon">❌</div>
        <div class="error-message">
            <h4>Error</h4>
            <p>{error_message}</p>
        </div>
    """

    if retry_action:
        error_html += f"""
        <div class="error-action">
            <button onclick="{retry_action}">
                🔄 Try Again
            </button>
        </div>
        """

    error_html += "</div>"
    return error_html


# Example usage and initialization
def get_default_api_client() -> APIClient:
    """Get default API client with environment configuration"""
    return create_api_client()


async def get_async_api_client() -> AsyncAPIClient:
    """Get default async API client with environment configuration"""
    return create_api_client(use_async=True)