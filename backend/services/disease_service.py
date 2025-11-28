"""
Disease information management service for Plant Disease Detection API
"""

import logging
import os
import json
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class DiseaseService:
    """
    Service for managing plant disease information and database operations
    """

    def __init__(self, database_path: Optional[str] = None):
        """Initialize disease service with database configuration"""
        self.database_path = database_path or os.getenv("DATABASE_URL", "./data/disease_database.json")
        self._diseases_cache = None
        self._categories_cache = None
        self._search_index = None

        # Load diseases on initialization
        self._load_diseases()

    def _load_diseases(self):
        """Load diseases from JSON database"""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self._diseases_cache = data.get('diseases', [])
                self._categories_cache = None
                self._build_search_index()

                logger.info(f"Loaded {len(self._diseases_cache)} diseases from database")
            else:
                logger.warning(f"Disease database not found at {self.database_path}")
                self._diseases_cache = []
                self._categories_cache = None
                self._search_index = None

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in disease database: {str(e)}")
            self._diseases_cache = []
        except Exception as e:
            logger.error(f"Error loading disease database: {str(e)}")
            self._diseases_cache = []

    def _build_search_index(self):
        """Build search index for faster disease lookup"""
        if not self._diseases_cache:
            self._search_index = {}
            return

        search_index = {}
        for disease in self._diseases_cache:
            disease_id = disease.get('id')
            if disease_id is None:
                continue

            # Build searchable text from multiple fields
            searchable_fields = [
                disease.get('name', ''),
                disease.get('scientific_name', ''),
                ' '.join(disease.get('symptoms', [])),
                ' '.join(disease.get('affected_plants', [])),
                ' '.join(disease.get('causes', [])),
                disease.get('category', ''),
                disease.get('severity', '')
            ]

            # Combine all searchable text
            combined_text = ' '.join(filter(None, searchable_fields)).lower()

            # Extract keywords (simple approach)
            keywords = re.findall(r'\b\w+\b', combined_text)
            keywords = [kw for kw in keywords if len(kw) > 2]  # Filter very short words

            # Build index
            search_index[disease_id] = {
                'disease': disease,
                'keywords': keywords,
                'searchable_text': combined_text
            }

        self._search_index = search_index
        logger.info(f"Built search index for {len(search_index)} diseases")

    def get_all_diseases(self) -> List[Dict[str, Any]]:
        """
        Get all diseases from the database

        Returns:
            List of disease dictionaries
        """
        return self._diseases_cache.copy() if self._diseases_cache else []

    def get_disease_by_id(self, disease_id: int) -> Optional[Dict[str, Any]]:
        """
        Get specific disease by ID

        Args:
            disease_id: Disease ID to search for

        Returns:
            Disease dictionary or None if not found
        """
        if not self._diseases_cache:
            return None

        for disease in self._diseases_cache:
            if disease.get('id') == disease_id:
                return disease.copy()

        return None

    def search_diseases(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search diseases by name, symptoms, or affected plants

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching disease dictionaries, sorted by relevance
        """
        if not self._search_index or not query:
            return []

        query_lower = query.lower()
        query_words = re.findall(r'\b\w+\b', query_lower)
        query_words = [qw for qw in query_words if len(qw) > 2]

        if not query_words:
            return []

        # Score each disease based on keyword matches
        scored_results = []

        for disease_id, index_data in self._search_index.items():
            keywords = index_data['keywords']
            searchable_text = index_data['searchable_text']
            disease = index_data['disease']

            # Calculate relevance score
            score = 0

            # Exact phrase match (highest score)
            if query_lower in searchable_text:
                score += 100

            # Individual word matches
            for word in query_words:
                if word in keywords:
                    score += 10

            # Partial matches
            for word in query_words:
                for keyword in keywords:
                    if word in keyword or keyword in word:
                        score += 2

            # Disease name match (bonus)
            disease_name = disease.get('name', '').lower()
            if query_lower in disease_name:
                score += 50

            if score > 0:
                scored_results.append({
                    'disease': disease,
                    'score': score
                })

        # Sort by score (descending)
        scored_results.sort(key=lambda x: x['score'], reverse=True)

        # Apply limit if specified
        if limit:
            scored_results = scored_results[:limit]

        # Return just the diseases
        return [result['disease'] for result in scored_results]

    def get_diseases_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get diseases filtered by category

        Args:
            category: Disease category to filter by

        Returns:
            List of disease dictionaries in the specified category
        """
        if not self._diseases_cache:
            return []

        category_lower = category.lower()
        filtered_diseases = []

        for disease in self._diseases_cache:
            disease_category = disease.get('category', '').lower()
            if disease_category == category_lower:
                filtered_diseases.append(disease)

        return filtered_diseases

    def get_diseases_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """
        Get diseases filtered by severity level

        Args:
            severity: Severity level to filter by (Low, Moderate, High, Critical)

        Returns:
            List of disease dictionaries with the specified severity
        """
        if not self._diseases_cache:
            return []

        severity_lower = severity.lower()
        filtered_diseases = []

        for disease in self._diseases_cache:
            disease_severity = disease.get('severity', '').lower()
            if disease_severity == severity_lower:
                filtered_diseases.append(disease)

        return filtered_diseases

    def get_diseases_by_plant(self, plant_name: str) -> List[Dict[str, Any]]:
        """
        Get diseases that affect a specific plant

        Args:
            plant_name: Name of the plant to search for

        Returns:
            List of disease dictionaries that affect the specified plant
        """
        if not self._diseases_cache:
            return []

        plant_lower = plant_name.lower()
        filtered_diseases = []

        for disease in self._diseases_cache:
            affected_plants = disease.get('affected_plants', [])
            for plant in affected_plants:
                if plant_lower in plant.lower() or plant.lower() in plant_lower:
                    filtered_diseases.append(disease)
                    break

        return filtered_diseases

    def get_categories(self) -> List[str]:
        """
        Get all unique disease categories

        Returns:
            List of category names
        """
        if self._categories_cache is not None:
            return self._categories_cache

        if not self._diseases_cache:
            self._categories_cache = []
            return []

        categories = set()
        for disease in self._diseases_cache:
            category = disease.get('category', '')
            if category:
                categories.add(category)

        self._categories_cache = sorted(list(categories))
        return self._categories_cache

    def get_severity_levels(self) -> List[str]:
        """
        Get all unique severity levels

        Returns:
            List of severity level names
        """
        if not self._diseases_cache:
            return []

        severities = set()
        for disease in self._diseases_cache:
            severity = disease.get('severity', '')
            if severity:
                severities.add(severity)

        return sorted(list(severities))

    def get_affected_plants(self) -> List[str]:
        """
        Get all unique affected plants

        Returns:
            List of plant names
        """
        if not self._diseases_cache:
            return []

        plants = set()
        for disease in self._diseases_cache:
            affected_plants = disease.get('affected_plants', [])
            for plant in affected_plants:
                if plant:
                    plants.add(plant)

        return sorted(list(plants))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dictionary with various statistics about the disease database
        """
        if not self._diseases_cache:
            return {
                "total_diseases": 0,
                "total_categories": 0,
                "total_severity_levels": 0,
                "total_affected_plants": 0,
                "contagious_diseases": 0,
                "database_last_updated": None
            }

        total_diseases = len(self._diseases_cache)
        categories = self.get_categories()
        severity_levels = self.get_severity_levels()
        affected_plants = self.get_affected_plants()

        contagious_count = sum(
            1 for disease in self._diseases_cache
            if disease.get('contagious', False)
        )

        # Get database modification time
        last_updated = None
        try:
            if os.path.exists(self.database_path):
                file_time = os.path.getmtime(self.database_path)
                last_updated = datetime.fromtimestamp(file_time).isoformat()
        except Exception as e:
            logger.error(f"Error getting database modification time: {str(e)}")

        return {
            "total_diseases": total_diseases,
            "total_categories": len(categories),
            "total_severity_levels": len(severity_levels),
            "total_affected_plants": len(affected_plants),
            "contagious_diseases": contagious_count,
            "categories": categories,
            "severity_levels": severity_levels,
            "database_last_updated": last_updated
        }

    def add_disease(self, disease_data: Dict[str, Any]) -> bool:
        """
        Add a new disease to the database

        Args:
            disease_data: Disease information dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._diseases_cache:
                self._diseases_cache = []

            # Validate required fields
            required_fields = ['name', 'scientific_name', 'category']
            for field in required_fields:
                if field not in disease_data:
                    logger.error(f"Missing required field: {field}")
                    return False

            # Generate unique ID
            existing_ids = [d.get('id', 0) for d in self._diseases_cache]
            new_id = max(existing_ids) + 1 if existing_ids else 1

            # Add metadata
            disease_data['id'] = new_id
            disease_data['created_at'] = datetime.utcnow().isoformat()

            # Add to cache
            self._diseases_cache.append(disease_data)
            self._categories_cache = None
            self._build_search_index()

            logger.info(f"Added disease: {disease_data.get('name')} (ID: {new_id})")
            return True

        except Exception as e:
            logger.error(f"Error adding disease: {str(e)}")
            return False

    def update_disease(self, disease_id: int, updates: Dict[str, Any]) -> bool:
        """
        Update an existing disease in the database

        Args:
            disease_id: ID of the disease to update
            updates: Dictionary of fields to update

        Returns:
            True if successful, False otherwise
        """
        try:
            for i, disease in enumerate(self._diseases_cache):
                if disease.get('id') == disease_id:
                    # Update fields
                    for key, value in updates.items():
                        if key != 'id':  # Don't allow ID changes
                            disease[key] = value

                    # Add update metadata
                    disease['updated_at'] = datetime.utcnow().isoformat()

                    # Clear caches
                    self._categories_cache = None
                    self._build_search_index()

                    logger.info(f"Updated disease ID {disease_id}: {disease.get('name')}")
                    return True

            logger.warning(f"Disease with ID {disease_id} not found")
            return False

        except Exception as e:
            logger.error(f"Error updating disease {disease_id}: {str(e)}")
            return False

    def delete_disease(self, disease_id: int) -> bool:
        """
        Delete a disease from the database

        Args:
            disease_id: ID of the disease to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            for i, disease in enumerate(self._diseases_cache):
                if disease.get('id') == disease_id:
                    disease_name = disease.get('name', 'Unknown')

                    # Remove from cache
                    del self._diseases_cache[i]
                    self._categories_cache = None
                    self._build_search_index()

                    logger.info(f"Deleted disease: {disease_name} (ID: {disease_id})")
                    return True

            logger.warning(f"Disease with ID {disease_id} not found")
            return False

        except Exception as e:
            logger.error(f"Error deleting disease {disease_id}: {str(e)}")
            return False

    def save_database(self, backup_path: Optional[str] = None) -> bool:
        """
        Save the current disease database to file

        Args:
            backup_path: Optional path to create backup

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup if requested
            if backup_path and os.path.exists(self.database_path):
                import shutil
                shutil.copy2(self.database_path, backup_path)
                logger.info(f"Created backup: {backup_path}")

            # Save current database
            database_data = {
                'diseases': self._diseases_cache,
                'metadata': {
                    'total_diseases': len(self._diseases_cache),
                    'last_updated': datetime.utcnow().isoformat(),
                    'version': '1.0'
                }
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)

            with open(self.database_path, 'w', encoding='utf-8') as f:
                json.dump(database_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Database saved to {self.database_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving database: {str(e)}")
            return False

    def validate_disease_data(self, disease_data: Dict[str, Any]) -> List[str]:
        """
        Validate disease data and return list of errors

        Args:
            disease_data: Disease information dictionary

        Returns:
            List of validation error messages
        """
        errors = []

        # Required fields
        required_fields = ['name', 'scientific_name', 'category']
        for field in required_fields:
            if not disease_data.get(field):
                errors.append(f"Missing required field: {field}")

        # Validate severity
        valid_severities = ['Low', 'Moderate', 'High', 'Critical']
        severity = disease_data.get('severity')
        if severity and severity not in valid_severities:
            errors.append(f"Invalid severity: {severity}. Valid options: {valid_severities}")

        # Validate lists
        list_fields = ['symptoms', 'causes', 'treatment', 'prevention', 'affected_plants']
        for field in list_fields:
            if field in disease_data:
                if not isinstance(disease_data[field], list):
                    errors.append(f"Field '{field}' must be a list")
                elif not all(isinstance(item, str) for item in disease_data[field]):
                    errors.append(f"All items in '{field}' must be strings")

        # Validate contagious flag
        if 'contagious' in disease_data and not isinstance(disease_data['contagious'], bool):
            errors.append("Field 'contagious' must be a boolean")

        return errors

    def get_similar_diseases(self, disease_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get diseases similar to the specified disease

        Args:
            disease_id: ID of the reference disease
            limit: Maximum number of similar diseases to return

        Returns:
            List of similar disease dictionaries
        """
        reference_disease = self.get_disease_by_id(disease_id)
        if not reference_disease:
            return []

        similar_diseases = []

        for disease in self._diseases_cache:
            if disease.get('id') == disease_id:
                continue

            similarity_score = 0

            # Category similarity
            if disease.get('category') == reference_disease.get('category'):
                similarity_score += 30

            # Severity similarity
            if disease.get('severity') == reference_disease.get('severity'):
                similarity_score += 20

            # Affected plants overlap
            ref_plants = set(reference_disease.get('affected_plants', []))
            disease_plants = set(disease.get('affected_plants', []))
            plants_overlap = len(ref_plants & disease_plants)
            if ref_plants:
                similarity_score += (plants_overlap / len(ref_plants)) * 25

            # Symptom similarity (simple keyword overlap)
            ref_symptoms = ' '.join(reference_disease.get('symptoms', [])).lower()
            disease_symptoms = ' '.join(disease.get('symptoms', [])).lower()
            symptom_words = set(ref_symptoms.split())
            disease_words = set(disease_symptoms.split())
            symptom_overlap = len(symptom_words & disease_words)
            if symptom_words:
                similarity_score += (symptom_overlap / len(symptom_words)) * 25

            if similarity_score > 0:
                similar_diseases.append({
                    'disease': disease,
                    'similarity_score': similarity_score
                })

        # Sort by similarity score and limit results
        similar_diseases.sort(key=lambda x: x['similarity_score'], reverse=True)
        return [item['disease'] for item in similar_diseases[:limit]]

    def reload_database(self):
        """Reload the disease database from file"""
        self._load_diseases()
        logger.info("Disease database reloaded from file")

    def get_database_hash(self) -> str:
        """
        Get a hash of the current database for cache invalidation

        Returns:
            MD5 hash string
        """
        if not self._diseases_cache:
            return ""

        # Create a deterministic string representation
        disease_strings = []
        for disease in sorted(self._diseases_cache, key=lambda x: x.get('id', 0)):
            disease_strings.append(json.dumps(disease, sort_keys=True))

        combined_data = ''.join(disease_strings)
        return hashlib.md5(combined_data.encode()).hexdigest()