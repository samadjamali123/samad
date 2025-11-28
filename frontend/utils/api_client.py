"""
API Client for communicating with the Plant Disease Detection Backend
"""

import os
import io
import json
import requests
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import base64
import time
from PIL import Image
import streamlit as st


class APIClient:
    """
    Client for interacting with the Plant Disease Detection API
    """

    def __init__(self, base_url: Optional[str] = None):
        """Initialize API client with backend URL"""
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        self.session = requests.Session()
        self.timeout = 30  # seconds
        self.max_retries = 3

        # Set default headers
        self.session.headers.update({
            'User-Agent': 'Plant-Detection-Streamlit-Client/1.0',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to the API

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: JSON data for request body
            files: Files for multipart upload
            params: URL parameters
            timeout: Request timeout override

        Returns:
            Response JSON dictionary
        """
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        timeout = timeout or self.timeout

        for attempt in range(self.max_retries):
            try:
                if files:
                    # Multipart form data
                    headers = {'Accept': 'application/json'}  # Remove content-type for multipart
                    response = self.session.request(
                        method=method,
                        url=url,
                        data=data,
                        files=files,
                        params=params,
                        headers=headers,
                        timeout=timeout
                    )
                else:
                    # JSON request
                    response = self.session.request(
                        method=method,
                        url=url,
                        json=data,
                        params=params,
                        timeout=timeout
                    )

                # Handle response
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    return {"success": False, "error": "Resource not found", "status_code": 404}
                elif response.status_code == 413:
                    return {"success": False, "error": "File too large", "status_code": 413}
                elif response.status_code == 422:
                    return {"success": False, "error": "Invalid request data", "status_code": 422}
                elif response.status_code == 500:
                    return {"success": False, "error": "Internal server error", "status_code": 500}
                elif response.status_code == 503:
                    return {"success": False, "error": "Service unavailable", "status_code": 503}
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "status_code": response.status_code
                    }

            except requests.exceptions.Timeout:
                if attempt == self.max_retries - 1:
                    return {"success": False, "error": "Request timeout"}
                time.sleep(1 * (attempt + 1))  # Exponential backoff

            except requests.exceptions.ConnectionError:
                if attempt == self.max_retries - 1:
                    return {"success": False, "error": "Cannot connect to API server"}
                time.sleep(1 * (attempt + 1))

            except requests.exceptions.RequestException as e:
                return {"success": False, "error": f"Request failed: {str(e)}"}

        return {"success": False, "error": "Max retries exceeded"}

    def health_check(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Check API health status

        Args:
            detailed: Whether to get detailed health information

        Returns:
            Health status dictionary
        """
        endpoint = "api/health/detailed" if detailed else "api/health"
        return self._make_request("GET", endpoint)

    def readiness_check(self) -> Dict[str, Any]:
        """
        Check API readiness status

        Returns:
            Readiness status dictionary
        """
        return self._make_request("GET", "api/health/ready")

    def detect_disease(
        self,
        image_file,
        confidence_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Detect disease from uploaded image

        Args:
            image_file: Uploaded file object from Streamlit
            confidence_threshold: Minimum confidence threshold

        Returns:
            Detection result dictionary
        """
        # Validate file
        if not image_file:
            return {"success": False, "error": "No image file provided"}

        # Check file size
        max_size = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB default
        if hasattr(image_file, 'size') and image_file.size > max_size:
            return {
                "success": False,
                "error": f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB"
            }

        # Prepare files for upload
        files = {"image": (image_file.name, image_file.getvalue(), image_file.type)}
        data = {"confidence_threshold": confidence_threshold}

        return self._make_request("POST", "api/detect", data=data, files=files)

    def detect_disease_batch(
        self,
        image_files: List,
        confidence_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Detect diseases from multiple images

        Args:
            image_files: List of uploaded file objects
            confidence_threshold: Minimum confidence threshold

        Returns:
            Batch detection result dictionary
        """
        if not image_files:
            return {"success": False, "error": "No image files provided"}

        max_batch_size = 10
        if len(image_files) > max_batch_size:
            return {
                "success": False,
                "error": f"Too many images. Maximum allowed: {max_batch_size}"
            }

        # Prepare files for upload
        files = {}
        for i, image_file in enumerate(image_files):
            if hasattr(image_file, 'getvalue'):
                files[f"images"] = (image_file.name, image_file.getvalue(), image_file.type)

        data = {"confidence_threshold": confidence_threshold}

        return self._make_request("POST", "api/detect/batch", data=data, files=files)

    def get_detection_status(self) -> Dict[str, Any]:
        """
        Get detection service status

        Returns:
            Detection service status dictionary
        """
        return self._make_request("GET", "api/detect/status")

    def get_all_diseases(self) -> Dict[str, Any]:
        """
        Get all available diseases

        Returns:
            Diseases dictionary
        """
        return self._make_request("GET", "api/diseases")

    def get_disease_by_id(self, disease_id: int) -> Dict[str, Any]:
        """
        Get specific disease by ID

        Args:
            disease_id: Disease ID

        Returns:
            Disease dictionary
        """
        return self._make_request("GET", f"api/diseases/{disease_id}")

    def search_diseases(
        self,
        query: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search diseases by query

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            Search results dictionary
        """
        params = {"q": query, "limit": limit}
        return self._make_request("GET", "api/diseases/search", params=params)

    def get_diseases_by_category(self, category: str) -> Dict[str, Any]:
        """
        Get diseases filtered by category

        Args:
            category: Disease category

        Returns:
            Diseases dictionary for category
        """
        return self._make_request("GET", f"api/diseases/category/{category}")

    def get_disease_categories(self) -> Dict[str, Any]:
        """
        Get all disease categories

        Returns:
            Categories dictionary
        """
        return self._make_request("GET", "api/diseases/categories")

    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Statistics dictionary
        """
        # Combine health check and diseases to get stats
        diseases_response = self.get_all_diseases()
        if not diseases_response.get("success"):
            return diseases_response

        diseases = diseases_response.get("diseases", [])
        stats = {
            "total_diseases": len(diseases),
            "categories": list(set(d.get("category", "") for d in diseases)),
            "severity_levels": list(set(d.get("severity", "") for d in diseases)),
            "affected_plants": list(set(plant for d in diseases for plant in d.get("affected_plants", []))),
            "contagious_count": sum(1 for d in diseases if d.get("contagious", False))
        }

        return {
            "success": True,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }

    def download_disease_image(self, image_url: str) -> Optional[bytes]:
        """
        Download disease image from static URL

        Args:
            image_url: Relative or absolute URL to image

        Returns:
            Image bytes or None if failed
        """
        try:
            # Convert relative URLs to absolute
            if image_url.startswith('/'):
                image_url = f"{self.base_url}{image_url}"

            response = self.session.get(image_url, timeout=10)
            if response.status_code == 200:
                return response.content
            else:
                return None

        except Exception as e:
            st.error(f"Error downloading image: {str(e)}")
            return None

    def get_image_as_base64(self, image_url: str) -> Optional[str]:
        """
        Get image as base64 string for display

        Args:
            image_url: URL to image

        Returns:
            Base64 string or None if failed
        """
        image_bytes = self.download_disease_image(image_url)
        if image_bytes:
            return base64.b64encode(image_bytes).decode()
        return None

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test API connection

        Returns:
            Tuple of (is_connected, message)
        """
        try:
            health_response = self.health_check()
            if health_response.get("success"):
                return True, "Connected successfully"
            else:
                return False, f"API error: {health_response.get('error', 'Unknown error')}"

        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def get_request_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get request history (simulated - would require backend storage in real implementation)

        Args:
            limit: Maximum number of history items

        Returns:
            List of request history items
        """
        # In a real implementation, this would query the backend
        # For now, return session-based history
        if hasattr(st.session_state, 'detection_history'):
            return st.session_state.detection_history[-limit:]
        return []

    def save_request_to_history(self, request_data: Dict[str, Any]):
        """
        Save request to session history

        Args:
            request_data: Request and response data
        """
        if 'detection_history' not in st.session_state:
            st.session_state.detection_history = []

        history_item = {
            "timestamp": datetime.now().isoformat(),
            "request_data": request_data,
            "id": len(st.session_state.detection_history) + 1
        }

        st.session_state.detection_history.append(history_item)

        # Keep only last 50 items in session
        if len(st.session_state.detection_history) > 50:
            st.session_state.detection_history = st.session_state.detection_history[-50:]

    def clear_request_history(self):
        """Clear request history"""
        if 'detection_history' in st.session_state:
            st.session_state.detection_history = []

    def get_api_info(self) -> Dict[str, Any]:
        """
        Get API information

        Returns:
            API information dictionary
        """
        try:
            # Get root endpoint
            response = self._make_request("GET", "")
            return response if response.get("success") else {
                "success": False,
                "error": "API info unavailable"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def validate_image_file(self, file) -> Tuple[bool, str]:
        """
        Validate uploaded image file

        Args:
            file: Uploaded file object

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file:
            return False, "No file provided"

        # Check file type
        allowed_extensions = os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png,webp").split(",")
        file_extension = file.name.lower().split('.')[-1] if '.' in file.name else ""

        if file_extension not in allowed_extensions:
            return False, f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"

        # Check file size
        max_size = int(os.getenv("MAX_FILE_SIZE", "10485760"))
        if hasattr(file, 'size') and file.size > max_size:
            return False, f"File too large. Maximum: {max_size / (1024*1024):.1f}MB"

        # Try to open as image
        try:
            Image.open(io.BytesIO(file.getvalue()))
            return True, ""
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"

    def format_error_message(self, error_response: Dict[str, Any]) -> str:
        """
        Format error message from API response

        Args:
            error_response: Error response from API

        Returns:
            Formatted error message
        """
        if not error_response:
            return "Unknown error occurred"

        error_msg = error_response.get("error", "Unknown error")
        status_code = error_response.get("status_code")

        if status_code == 413:
            return f"📁 **File Too Large**: {error_msg}"
        elif status_code == 422:
            return f"📝 **Invalid Data**: {error_msg}"
        elif status_code == 500:
            return f"🔧 **Server Error**: The service encountered an error. Please try again."
        elif status_code == 503:
            return f"⚠️ **Service Unavailable**: The detection service is temporarily down."
        else:
            return f"❌ **Error**: {error_msg}"

    def format_detection_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format detection result for display

        Args:
            result: Raw detection result from API

        Returns:
            Formatted result dictionary
        """
        if not result.get("success"):
            return {
                "success": False,
                "error": self.format_error_message(result),
                "timestamp": datetime.now().isoformat()
            }

        prediction = result.get("prediction", {})
        image_info = result.get("image_info", {})

        formatted_result = {
            "success": True,
            "disease_name": prediction.get("disease_name", "Unknown"),
            "scientific_name": prediction.get("scientific_name", ""),
            "confidence": prediction.get("confidence", 0.0),
            "severity": prediction.get("severity", "Unknown"),
            "class_index": prediction.get("class_index", 0),
            "detected_at": prediction.get("detected_at", datetime.now().isoformat()),
            "image_info": {
                "filename": image_info.get("filename", ""),
                "size": image_info.get("size", 0),
                "dimensions": image_info.get("dimensions", []),
                "format": image_info.get("format", "")
            },
            "processing_time": result.get("processing_time", 0.0),
            "timestamp": datetime.now().isoformat()
        }

        # Format file size for display
        if formatted_result["image_info"]["size"]:
            size_bytes = formatted_result["image_info"]["size"]
            if size_bytes < 1024:
                formatted_result["image_info"]["size_display"] = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                formatted_result["image_info"]["size_display"] = f"{size_bytes / 1024:.1f} KB"
            else:
                formatted_result["image_info"]["size_display"] = f"{size_bytes / (1024*1024):.1f} MB"

        return formatted_result