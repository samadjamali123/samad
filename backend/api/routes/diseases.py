"""
Disease information endpoints for the Plant Disease Detection API
"""

import logging
import os
import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# Placeholder for disease service - will be implemented later
class DiseaseService:
    def __init__(self):
        self.db_path = os.getenv("DATABASE_URL", "./data/disease_database.json")
        self._diseases_cache = None

    def load_diseases(self):
        """Load diseases from JSON database"""
        if self._diseases_cache is None:
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self._diseases_cache = data.get('diseases', [])
                logger.info(f"Loaded {len(self._diseases_cache)} diseases from database")
            except FileNotFoundError:
                logger.warning(f"Disease database not found at {self.db_path}")
                self._diseases_cache = []
            except Exception as e:
                logger.error(f"Error loading disease database: {str(e)}")
                self._diseases_cache = []
        return self._diseases_cache

    def get_all_diseases(self) -> List[Dict]:
        """Get all diseases"""
        return self.load_diseases()

    def get_disease_by_id(self, disease_id: int) -> Optional[Dict]:
        """Get specific disease by ID"""
        diseases = self.load_diseases()
        for disease in diseases:
            if disease.get('id') == disease_id:
                return disease
        return None

    def search_diseases(self, query: str) -> List[Dict]:
        """Search diseases by name or symptoms"""
        diseases = self.load_diseases()
        query_lower = query.lower()
        results = []

        for disease in diseases:
            # Search in name, scientific name, symptoms
            searchable_text = [
                disease.get('name', ''),
                disease.get('scientific_name', ''),
                ' '.join(disease.get('symptoms', [])),
                ' '.join(disease.get('affected_plants', []))
            ]

            if any(query_lower in text.lower() for text in searchable_text):
                results.append(disease)

        return results

    def get_diseases_by_category(self, category: str) -> List[Dict]:
        """Get diseases filtered by category"""
        diseases = self.load_diseases()
        return [d for d in diseases if d.get('category', '').lower() == category.lower()]

    def get_categories(self) -> List[str]:
        """Get all disease categories"""
        diseases = self.load_diseases()
        categories = set()
        for disease in diseases:
            if disease.get('category'):
                categories.add(disease['category'])
        return sorted(list(categories))

# Initialize service
disease_service = DiseaseService()


# Pydantic models for response
class DiseaseResponse(BaseModel):
    id: int
    name: str
    scientific_name: str
    category: str
    severity: str
    contagious: bool
    affected_plants: List[str]
    symptoms: List[str]
    causes: List[str]
    treatment: List[str]
    prevention: List[str]
    image_urls: List[str] = Field(default_factory=list)


class DiseaseListResponse(BaseModel):
    success: bool = True
    diseases: List[DiseaseResponse]
    count: int


class DiseaseDetailResponse(BaseModel):
    success: bool = True
    disease: DiseaseResponse


class CategoryResponse(BaseModel):
    success: bool = True
    categories: List[str]
    count: int


@router.get("/diseases", response_model=DiseaseListResponse, summary="Get All Diseases")
async def get_all_diseases():
    """
    Retrieve all available plant diseases from the database
    """
    try:
        diseases = disease_service.get_all_diseases()
        disease_responses = [DiseaseResponse(**disease) for disease in diseases]

        return DiseaseListResponse(
            diseases=disease_responses,
            count=len(disease_responses)
        )
    except Exception as e:
        logger.error(f"Error fetching all diseases: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve diseases")


@router.get("/diseases/{disease_id}", response_model=DiseaseDetailResponse, summary="Get Disease Details")
async def get_disease_by_id(
    disease_id: int = Path(..., ge=1, description="Disease ID to retrieve")
):
    """
    Retrieve detailed information about a specific disease
    """
    try:
        disease = disease_service.get_disease_by_id(disease_id)

        if not disease:
            raise HTTPException(status_code=404, detail=f"Disease with ID {disease_id} not found")

        return DiseaseDetailResponse(disease=DiseaseResponse(**disease))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching disease {disease_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve disease {disease_id}")


@router.get("/diseases/search", response_model=DiseaseListResponse, summary="Search Diseases")
async def search_diseases(
    q: str = Query(..., min_length=2, description="Search query for diseases"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results")
):
    """
    Search diseases by name, symptoms, or affected plants
    """
    try:
        diseases = disease_service.search_diseases(q)
        limited_diseases = diseases[:limit]
        disease_responses = [DiseaseResponse(**disease) for disease in limited_diseases]

        return DiseaseListResponse(
            diseases=disease_responses,
            count=len(disease_responses)
        )

    except Exception as e:
        logger.error(f"Error searching diseases with query '{q}': {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search diseases")


@router.get("/diseases/category/{category}", response_model=DiseaseListResponse, summary="Get Diseases by Category")
async def get_diseases_by_category(
    category: str = Path(..., description="Disease category to filter by")
):
    """
    Retrieve diseases filtered by category (Fungal, Bacterial, Viral, etc.)
    """
    try:
        diseases = disease_service.get_diseases_by_category(category)
        disease_responses = [DiseaseResponse(**disease) for disease in diseases]

        return DiseaseListResponse(
            diseases=disease_responses,
            count=len(disease_responses)
        )

    except Exception as e:
        logger.error(f"Error fetching diseases for category '{category}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve diseases for category {category}")


@router.get("/diseases/categories", response_model=CategoryResponse, summary="Get Disease Categories")
async def get_disease_categories():
    """
    Retrieve all available disease categories
    """
    try:
        categories = disease_service.get_categories()

        return CategoryResponse(
            categories=categories,
            count=len(categories)
        )

    except Exception as e:
        logger.error(f"Error fetching disease categories: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve disease categories")