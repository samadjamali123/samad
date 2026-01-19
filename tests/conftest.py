"""
Pytest configuration and fixtures for Plant Leaf Disease Detection System testing.

This module provides common test fixtures, configuration, and utilities
for unit tests, integration tests, and end-to-end tests.
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from PIL import Image
import numpy as np

from backend.main import app
from backend.services.config import Settings
from backend.services.ai_service import AIService
from backend.services.disease_service import DiseaseService
from backend.services.image_processor import ImageProcessor
from backend.services.monitoring_service import MonitoringService
from backend.services.database_service import DatabaseService

# Test logger
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_settings():
    """Create test settings for all tests."""
    return Settings(
        ENVIRONMENT="testing",
        DEBUG=True,
        
        # API Settings
        API_HOST="127.0.0.1",
        API_PORT=8001,
        API_PREFIX="/api/v1",
        
        # File upload settings
        MAX_FILE_SIZE_MB=5,
        MIN_IMAGE_SIZE=100,
        MAX_CONCURRENT_UPLOADS=3,
        
        # AI Model Settings
        GROK_API_KEY="test_grok_key",
        PLANTNET_API_KEY="test_plantnet_key",
        INATURALIST_API_KEY="test_inaturalist_key",
        
        # Database Settings
        POSTGRES_URL="postgresql://test:test@localhost:5432/test_plantdetection",
        REDIS_URL="redis://localhost:6379/1",
        
        # Testing specific settings
        ENABLE_TESTING_MODE=True,
        MOCK_EXTERNAL_APIS=True,
        
        # Logging
        LOG_LEVEL="DEBUG",
        ENABLE_SQL_LOGGING=False,
        
        # Rate limiting
        MAX_REQUESTS_PER_MINUTE=100,
        
        # Feature flags
        ENABLE_ADVANCED_ANALYSIS=True,
        ENABLE_TREATMENT_CALCULATOR=True,
        ENABLE_RESULT_SHARING=True
    )


@pytest_asyncio.fixture
async def temp_test_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing."""
    # Create a simple RGB image
    image = Image.new('RGB', (300, 300), color='green')
    image_array = np.array(image)
    
    # Add some texture to simulate a leaf
    noise = np.random.randint(0, 50, (300, 300, 3), dtype=np.uint8)
    image_array = np.clip(image_array.astype(int) + noise - 25, 0, 255).astype(np.uint8)
    
    # Add some brown spots to simulate disease
    for _ in range(5):
        x, y = np.random.randint(50, 250, 2)
        image_array[x-10:x+10, y-10:y+10] = [139, 69, 19]  # Brown color
    
    final_image = Image.fromarray(image_array)
    
    # Convert to bytes
    from io import BytesIO
    buffer = BytesIO()
    final_image.save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()


@pytest.fixture
def sample_image_file(sample_image_bytes):
    """Create a temporary image file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_file.write(sample_image_bytes)
        temp_file.flush()
        yield temp_file.name
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.fixture
def multiple_sample_images():
    """Create multiple sample images for batch testing."""
    images = []
    for i in range(3):
        # Create different colored images
        colors = ['green', 'lightgreen', 'darkgreen']
        image = Image.new('RGB', (300, 300), color=colors[i])
        
        # Add random texture
        image_array = np.array(image)
        noise = np.random.randint(0, 30, (300, 300, 3), dtype=np.uint8)
        image_array = np.clip(image_array.astype(int) + noise - 15, 0, 255).astype(np.uint8)
        
        final_image = Image.fromarray(image_array)
        
        from io import BytesIO
        buffer = BytesIO()
        final_image.save(buffer, format='JPEG', quality=90)
        images.append(buffer.getvalue())
    
    return images


@pytest_asyncio.fixture
async def mock_ai_service(test_settings):
    """Create a mock AI service for testing."""
    service = AsyncMock(spec=AIService)
    
    # Mock available models
    service.get_available_models.return_value = ["grok", "plantnet", "inaturalist"]
    
    # Mock model health checks
    service.check_grok_health.return_value = True
    service.check_plantnet_health.return_value = True
    service.check_inaturalist_health.return_value = True
    
    # Mock analysis results
    mock_results = [
        {
            'model_type': 'grok',
            'success': True,
            'plant_identification': {
                'plant_name': 'tomato',
                'confidence': 0.92,
                'common_names': ['tomato', 'garden tomato']
            },
            'disease_identification': {
                'disease_name': 'early_blight',
                'confidence': 0.85,
                'symptoms': ['leaf spots', 'yellowing']
            },
            'processing_time_ms': 1500
        },
        {
            'model_type': 'plantnet',
            'success': True,
            'plant_identification': {
                'plant_name': 'tomato',
                'confidence': 0.88,
                'common_names': ['tomato']
            },
            'disease_identification': None,
            'processing_time_ms': 1200
        },
        {
            'model_type': 'inaturalist',
            'success': True,
            'plant_identification': {
                'plant_name': 'tomato',
                'confidence': 0.90,
                'common_names': ['tomato']
            },
            'disease_identification': None,
            'processing_time_ms': 1000
        }
    ]
    
    service.analyze_with_all_models.return_value = mock_results
    
    # Mock aggregation
    mock_aggregated = MagicMock()
    mock_aggregated.primary_plant_name = 'tomato'
    mock_aggregated.plant_confidence = 0.90
    mock_aggregated.plant_alternatives = ['potato', 'eggplant']
    mock_aggregated.primary_disease_name = 'early_blight'
    mock_aggregated.disease_confidence = 0.85
    mock_aggregated.disease_alternatives = []
    
    service.aggregate_results.return_value = mock_aggregated
    
    # Mock statistics
    service.get_model_statistics.return_value = {
        'total_requests': 1000,
        'successful_requests': 950,
        'average_processing_time': 1234,
        'model_success_rates': {
            'grok': 0.95,
            'plantnet': 0.92,
            'inaturalist': 0.89
        }
    }
    
    service.initialize.return_value = None
    service.cleanup.return_value = None
    
    return service


@pytest_asyncio.fixture
async def mock_disease_service():
    """Create a mock disease service for testing."""
    service = AsyncMock(spec=DiseaseService)
    
    # Mock health check
    service.health_check.return_value = True
    
    # Mock database stats
    mock_stats = MagicMock()
    mock_stats.total_diseases = 50
    mock_stats.total_plants = 25
    mock_stats.total_treatments = 120
    
    service.get_database_stats.return_value = mock_stats
    
    # Mock supported plants
    service.get_supported_plants.return_value = [
        'tomato', 'potato', 'apple', 'grape', 'corn',
        'wheat', 'rice', 'soybean', 'cotton', 'sugarcane'
    ]
    
    # Mock disease categories
    service.get_disease_categories.return_value = {
        'vegetables': ['tomato', 'potato', 'pepper'],
        'fruits': ['apple', 'grape', 'strawberry'],
        'grains': ['corn', 'wheat', 'rice'],
        'legumes': ['soybean', 'beans', 'peas']
    }
    
    # Mock disease search
    mock_disease = MagicMock()
    mock_disease.disease_id = 'early_blight_tomato'
    mock_disease.plant_name = 'tomato'
    mock_disease.disease_name = 'Early Blight'
    mock_disease.scientific_name = 'Alternaria solani'
    mock_disease.description = 'A common fungal disease affecting tomato plants...'
    mock_disease.severity = 'moderate'
    mock_disease.transmission = 'airborne, water splash'
    mock_disease.favorable_conditions = 'warm, humid weather'
    mock_disease.reference_images = ['images/early_blight_1.jpg', 'images/early_blight_2.jpg']
    
    service.get_disease_by_id.return_value = mock_disease
    
    service.initialize.return_value = None
    service.cleanup.return_value = None
    
    return service


# Test data generators
class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def create_test_image(width: int = 300, height: int = 300, 
                        color: str = 'green', format: str = 'JPEG') -> bytes:
        """Create a test image with specified parameters."""
        from io import BytesIO
        
        image = Image.new('RGB', (width, height), color=color)
        
        # Add some noise for realism
        image_array = np.array(image)
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        image_array = np.clip(image_array.astype(int) + noise - 15, 0, 255).astype(np.uint8)
        
        final_image = Image.fromarray(image_array)
        
        buffer = BytesIO()
        final_image.save(buffer, format=format, quality=95)
        return buffer.getvalue()
    
    @staticmethod
    def create_test_analysis_data(plant_name: str = 'tomato', 
                               disease_name: str = 'early_blight') -> Dict[str, Any]:
        """Create test analysis data."""
        return {
            'session_id': f"test_{uuid.uuid4().hex[:8]}",
            'plant_detection': {
                'plant_name': plant_name,
                'confidence': np.random.uniform(0.8, 0.95)
            },
            'disease_detection': {
                'primary_disease': {
                    'disease_name': disease_name,
                    'confidence': np.random.uniform(0.7, 0.9)
                } if disease_name else None
            },
            'metadata': {
                'image_quality_score': np.random.uniform(0.7, 0.95),
                'processing_time_total_ms': np.random.randint(1000, 5000)
            }
        }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "external_api: mark test that requires external API access"
    )