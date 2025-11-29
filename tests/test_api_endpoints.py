"""
Test API endpoints for Plant Leaf Disease Detection System.

This module contains comprehensive tests for all API endpoints including
health checks, image analysis, disease information, and monitoring.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    @pytest.mark.unit
    def test_root_endpoint(self, test_client: TestClient):
        """Test root endpoint returns basic API information."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "Plant Leaf Disease Detector API"
        assert "version" in data
        assert "description" in data
        assert "docs_url" in data
        assert "health_check" in data
    
    @pytest.mark.unit
    def test_health_check_endpoint(self, test_client: TestClient):
        """Test health check endpoint returns service status."""
        response = test_client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "timestamp" in data
        assert "services" in data
        assert "version" in data
        assert "uptime_seconds" in data
    
    @pytest.mark.unit
    def test_database_stats_endpoint(self, test_client: TestClient):
        """Test database statistics endpoint."""
        response = test_client.get("/api/v1/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_diseases" in data
        assert "total_plants" in data
        assert "total_treatments" in data
    
    @pytest.mark.unit
    def test_configuration_endpoint(self, test_client: TestClient):
        """Test configuration endpoint returns public settings."""
        response = test_client.get("/api/v1/config")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "max_file_size_mb" in data
        assert "supported_formats" in data
        assert "min_image_size" in data
        assert "enabled_features" in data


class TestAnalysisEndpoints:
    """Test image analysis endpoints."""
    
    @pytest.mark.integration
    def test_analyze_image_success(self, test_client: TestClient, sample_image_bytes):
        """Test successful image analysis."""
        response = test_client.post(
            "/api/v1/analyze",
            files={"image": ("test_leaf.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "session_id" in data
        assert "timestamp" in data
        assert "results" in data
        assert "processing_time_ms" in data
        
        # Check analysis results structure
        results = data["results"]
        assert "plant_detection" in results
        assert "disease_detection" in results
        assert "ai_model_results" in results
        assert "metadata" in results
    
    @pytest.mark.unit
    def test_analyze_image_invalid_format(self, test_client: TestClient):
        """Test image analysis with invalid file format."""
        invalid_file = b"This is not an image file"
        
        response = test_client.post(
            "/api/v1/analyze",
            files={"image": ("test.txt", invalid_file, "text/plain")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data or "detail" in data
    
    @pytest.mark.unit
    def test_analyze_image_no_file(self, test_client: TestClient):
        """Test image analysis without file."""
        response = test_client.post("/api/v1/analyze")
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.integration
    def test_analyze_image_with_quality_threshold(self, test_client: TestClient, sample_image_bytes):
        """Test image analysis with quality threshold parameter."""
        response = test_client.post(
            "/api/v1/analyze",
            files={"image": ("test_leaf.jpg", sample_image_bytes, "image/jpeg")},
            data={"quality_threshold": "0.8"}
        )
        
        assert response.status_code == 200
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_batch_analyze_images(self, test_client: TestClient, multiple_sample_images):
        """Test batch image analysis."""
        files = []
        for i, image_bytes in enumerate(multiple_sample_images):
            files.append(("images", f"test_leaf_{i}.jpg", image_bytes))
        
        response = test_client.post("/api/v1/analyze/batch", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "batch_id" in data
        assert "total_images" in data
        assert "successful_analyses" in data
        assert "failed_analyses" in data
        assert "individual_results" in data
    
    @pytest.mark.unit
    def test_batch_analyze_too_many_images(self, test_client: TestClient, multiple_sample_images):
        """Test batch analysis with too many images."""
        # Create more than allowed images (11 > 10)
        files = [("images", f"test_{i}.jpg", b"fake_image") for i in range(11)]
        
        response = test_client.post("/api/v1/analyze/batch", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "Maximum 10 images allowed" in str(data)
    
    @pytest.mark.integration
    def test_grad_cam_analysis(self, test_client: TestClient, sample_image_bytes):
        """Test Grad-CAM visualization generation."""
        response = test_client.post(
            "/api/v1/analyze/grad-cam",
            files={"image": ("test_leaf.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "session_id" in data
        assert "grad_cam_data" in data
        assert "processing_time_ms" in data


class TestDiseaseEndpoints:
    """Test disease information endpoints."""
    
    @pytest.mark.unit
    def test_get_disease_info_success(self, test_client: TestClient):
        """Test getting disease information for valid disease ID."""
        response = test_client.get("/api/v1/diseases/early_blight_tomato")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "disease" in data
        assert "treatments" in data
        
        disease = data["disease"]
        assert disease["disease_id"] == "early_blight_tomato"
        assert disease["plant_name"] == "tomato"
        assert "disease_name" in disease
        assert "scientific_name" in disease
        assert "description" in disease
    
    @pytest.mark.unit
    def test_get_disease_info_not_found(self, test_client: TestClient):
        """Test getting disease information for non-existent disease."""
        response = test_client.get("/api/v1/diseases/nonexistent_disease")
        
        assert response.status_code == 404
    
    @pytest.mark.unit
    def test_search_diseases_by_query(self, test_client: TestClient):
        """Test disease search by query string."""
        response = test_client.get("/api/v1/search/diseases?q=blight")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "results" in data
        assert "total_count" in data
        assert "query" in data
        
        # Check result structure
        if data["results"]:
            result = data["results"][0]
            assert "disease_id" in result
            assert "plant_name" in result
            assert "disease_name" in result
            assert "description" in result
    
    @pytest.mark.unit
    def test_search_diseases_by_plant(self, test_client: TestClient):
        """Test disease search by plant name."""
        response = test_client.get("/api/v1/search/diseases?plant=tomato")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "results" in data
        assert "plant" in data
    
    @pytest.mark.unit
    def test_search_diseases_no_parameters(self, test_client: TestClient):
        """Test disease search without required parameters."""
        response = test_client.get("/api/v1/search/diseases")
        
        assert response.status_code == 400
    
    @pytest.mark.unit
    def test_search_diseases_limit(self, test_client: TestClient):
        """Test disease search with limit parameter."""
        response = test_client.get("/api/v1/search/diseases?q=blight&limit=5")
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["results"]) <= 5
    
    @pytest.mark.unit
    def test_get_supported_plants(self, test_client: TestClient):
        """Test getting list of supported plants."""
        response = test_client.get("/api/v1/plants/supported")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "plants" in data
        assert "total_count" in data
        assert isinstance(data["plants"], list)
    
    @pytest.mark.unit
    def test_get_disease_categories(self, test_client: TestClient):
        """Test getting disease categories."""
        response = test_client.get("/api/v1/diseases/categories")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "categories" in data
        assert "total_categories" in data
        assert "total_diseases" in data
        
        # Check category structure
        categories = data["categories"]
        assert "vegetables" in categories
        assert "fruits" in categories
        assert "grains" in categories
        assert "legumes" in categories


class TestMonitoringEndpoints:
    """Test monitoring and analytics endpoints."""
    
    @pytest.mark.unit
    def test_get_monitoring_dashboard(self, test_client: TestClient):
        """Test getting monitoring dashboard data."""
        response = test_client.get("/api/v1/monitoring/dashboard")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "timestamp" in data
        assert "time_range_hours" in data
        assert "dashboard" in data
        
        dashboard = data["dashboard"]
        assert "system_metrics" in dashboard
        assert "model_metrics" in dashboard
        assert "business_metrics" in dashboard
    
    @pytest.mark.unit
    def test_get_system_metrics(self, test_client: TestClient):
        """Test getting current system metrics."""
        response = test_client.get("/api/v1/monitoring/system")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "timestamp" in data
        assert "system_metrics" in data
        
        metrics = data["system_metrics"]
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "disk_percent" in metrics
        assert "network_sent_mb" in metrics
        assert "network_recv_mb" in metrics
    
    @pytest.mark.unit
    def test_get_business_metrics(self, test_client: TestClient):
        """Test getting business analytics metrics."""
        response = test_client.get("/api/v1/monitoring/business")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "time_range_hours" in data
        assert "business_metrics" in data
    
    @pytest.mark.unit
    def test_get_alerts(self, test_client: TestClient):
        """Test getting monitoring alerts."""
        response = test_client.get("/api/v1/monitoring/alerts")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "alerts" in data
        assert "total_count" in data
        assert "filters" in data
    
    @pytest.mark.unit
    def test_get_alerts_with_filters(self, test_client: TestClient):
        """Test getting alerts with severity and limit filters."""
        response = test_client.get("/api/v1/monitoring/alerts?severity=critical&limit=10")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["filters"]["severity"] == "critical"
        assert data["filters"]["limit"] == 10
    
    @pytest.mark.unit
    def test_get_performance_report(self, test_client: TestClient):
        """Test getting performance report."""
        response = test_client.get("/api/v1/monitoring/reports/performance")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "time_range_hours" in data
        assert "report" in data
    
    @pytest.mark.unit
    def test_get_enhanced_health_check(self, test_client: TestClient):
        """Test enhanced health check with monitoring data."""
        response = test_client.get("/api/v1/monitoring/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "uptime" in data
        assert "environment" in data
        assert "system_metrics" in data
        assert "services" in data
        assert "recommendations" in data
    
    @pytest.mark.unit
    def test_export_monitoring_data(self, test_client: TestClient):
        """Test exporting monitoring data."""
        response = test_client.get("/api/v1/monitoring/export")
        
        assert response.status_code == 200
        # Check that response contains file download headers
        content_disposition = response.headers.get("content-disposition", "")
        assert "attachment" in content_disposition.lower()
        assert "monitoring_export" in content_disposition


class TestModelEndpoints:
    """Test AI model status and management endpoints."""
    
    @pytest.mark.unit
    def test_get_models_status(self, test_client: TestClient):
        """Test getting AI models status."""
        response = test_client.get("/api/v1/models/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "available_models" in data
        assert "health_status" in data
        assert "statistics" in data
        assert "total_models" in data
    
    @pytest.mark.unit
    def test_get_model_performance(self, test_client: TestClient):
        """Test getting performance metrics for specific model."""
        response = test_client.get("/api/v1/monitoring/models/grok")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "model_name" in data
        assert "time_range_hours" in data
        assert "metrics" in data


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.unit
    def test_rate_limiting(self, test_client: TestClient, sample_image_bytes):
        """Test rate limiting on analysis endpoint."""
        # Make multiple rapid requests to test rate limiting
        responses = []
        for _ in range(70):  # Exceed the limit of 60/minute
            response = test_client.post(
                "/api/v1/analyze",
                files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
            )
            responses.append(response)
            
            if response.status_code == 429:
                break
        
        # Should eventually hit rate limit
        rate_limited = any(r.status_code == 429 for r in responses)
        assert rate_limited, "Rate limiting should be triggered"
    
    @pytest.mark.unit
    def test_invalid_json_payload(self, test_client: TestClient):
        """Test handling of invalid JSON payloads."""
        response = test_client.post(
            "/api/v1/analyze",
            json="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    @pytest.mark.unit
    def test_missing_required_headers(self, test_client: TestClient):
        """Test handling of missing required headers where applicable."""
        response = test_client.post(
            "/api/v1/analyze",
            data="",
            headers={"Content-Type": "application/json"}  # Missing required content
        )
        
        # Should handle gracefully
        assert response.status_code in [400, 422]
    
    @pytest.mark.unit
    def test oversized_file(self, test_client: TestClient):
        """Test handling of oversized files."""
        # Create a large file (simulate oversized)
        large_file = b"x" * (6 * 1024 * 1024)  # 6MB (exceeds 5MB limit)
        
        response = test_client.post(
            "/api/v1/analyze",
            files={"image": ("large.jpg", large_file, "image/jpeg")}
        )
        
        assert response.status_code == 413  # Request Entity Too Large
        data = response.json()
        assert "too large" in str(data).lower()


class TestAsyncEndpoints:
    """Test async client functionality."""
    
    @pytest.mark.integration
    async def test_async_analyze_image(self, async_client: AsyncClient, sample_image_bytes):
        """Test async image analysis."""
        files = {"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
        
        response = await async_client.post("/api/v1/analyze", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @pytest.mark.integration
    async def test_async_health_check(self, async_client: AsyncClient):
        """Test async health check."""
        response = await async_client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


# Test utilities
class TestUtils:
    """Test utility functions and helpers."""
    
    @pytest.mark.unit
    def test_test_data_generator_image_creation(self):
        """Test TestDataGenerator creates valid images."""
        from tests.conftest import TestDataGenerator
        
        image_bytes = TestDataGenerator.create_test_image()
        assert len(image_bytes) > 0
        
        # Verify it's a valid JPEG
        from PIL import Image
        from io import BytesIO
        
        image = Image.open(BytesIO(image_bytes))
        assert image.format == 'JPEG'
        assert image.size == (300, 300)
    
    @pytest.mark.unit
    def test_test_data_generator_analysis_data(self):
        """Test TestDataGenerator creates valid analysis data."""
        from tests.conftest import TestDataGenerator
        
        data = TestDataGenerator.create_test_analysis_data()
        
        assert "session_id" in data
        assert "plant_detection" in data
        assert "disease_detection" in data
        assert "metadata" in data
        
        plant_detection = data["plant_detection"]
        assert "plant_name" in plant_detection
        assert "confidence" in plant_detection
        assert 0 <= plant_detection["confidence"] <= 1