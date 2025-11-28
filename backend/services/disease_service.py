"""
Disease Database Service

This module manages the comprehensive disease database including diseases,
treatments, and reference images for all major crops and their top
damaging diseases.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
import hashlib
from enum import Enum

from backend.models.disease import (
    Disease, TreatmentInfo, TreatmentMethod, TreatmentCategory,
    ProfessionalConsultation, EnvironmentalFactors, SeverityClassification,
    DatabaseStats, DiseaseInfoResponse
)
from backend.models.analysis import ProcessingError
from backend.services.config import Settings

logger = logging.getLogger(__name__)


class CropType(str, Enum):
    """Major crop categories"""
    VEGETABLES = "vegetables"
    FRUITS = "fruits"
    GRAINS = "grains"
    LEGUMES = "legumes"
    OTHER = "other"


class DiseaseService:
    """Comprehensive disease database management service"""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize disease service with settings"""
        self.settings = settings or Settings()
        self.diseases_data: Dict[str, Disease] = {}
        self.treatments_data: Dict[str, TreatmentInfo] = {}
        self.reference_images_map: Dict[str, List[str]] = {}
        self.plant_index: Dict[str, List[str]] = {}  # Plant name -> disease IDs
        self.symptom_index: Dict[str, List[str]] = {}  # Symptom -> disease IDs
        self.cause_index: Dict[str, List[str]] = {}  # Cause -> disease IDs
        self.last_updated = None

        # Database paths
        self.db_paths = self.settings.DATABASE_PATHS_ABSOLUTE
        self.diseases_file = Path(self.db_paths['diseases'])
        self.treatments_file = Path(self.db_paths['treatments'])
        self.reference_images_dir = Path(self.db_paths['reference_images'])

        logger.info(f"DiseaseService initialized with database paths: {self.db_paths}")

    async def initialize(self):
        """Initialize disease database by loading data files"""
        start_time = time.time()

        try:
            # Create necessary directories
            self.reference_images_dir.mkdir(parents=True, exist_ok=True)
            self.diseases_file.parent.mkdir(parents=True, exist_ok=True)
            self.treatments_file.parent.mkdir(parents=True, exist_ok=True)

            # Load or create disease database
            await self.load_or_create_diseases()

            # Load or create treatments database
            await self.load_or_create_treatments()

            # Build search indices
            self.build_search_indices()

            # Verify reference images
            await self.verify_reference_images()

            self.last_updated = time.time()
            load_time = time.time() - start_time

            logger.info(f"Disease database initialized successfully in {load_time:.2f}s")
            logger.info(f"Loaded {len(self.diseases_data)} diseases and {len(self.treatments_data)} treatment entries")

        except Exception as e:
            logger.error(f"Failed to initialize disease database: {str(e)}")
            raise

    async def load_or_create_diseases(self):
        """Load diseases from file or create comprehensive database"""
        if self.diseases_file.exists():
            try:
                with open(self.diseases_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convert to Disease objects
                for disease_data in data:
                    disease = Disease(**disease_data)
                    self.diseases_data[disease.disease_id] = disease

                logger.info(f"Loaded {len(self.diseases_data)} diseases from file")
                return

            except Exception as e:
                logger.warning(f"Failed to load diseases file: {str(e)}. Creating new database.")

        # Create comprehensive disease database
        await self.create_comprehensive_disease_database()

    async def load_or_create_treatments(self):
        """Load treatments from file or create comprehensive database"""
        if self.treatments_file.exists():
            try:
                with open(self.treatments_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convert to TreatmentInfo objects
                for treatment_data in data:
                    treatment = TreatmentInfo(**treatment_data)
                    self.treatments_data[treatment.disease_id] = treatment

                logger.info(f"Loaded {len(self.treatments_data)} treatment entries from file")
                return

            except Exception as e:
                logger.warning(f"Failed to load treatments file: {str(e)}. Creating new database.")

        # Create comprehensive treatment database
        await self.create_comprehensive_treatment_database()

    async def create_comprehensive_disease_database(self):
        """Create comprehensive disease database for major crops"""
        logger.info("Creating comprehensive disease database...")

        diseases = []

        # TOMATO DISEASES
        tomato_diseases = [
            {
                "disease_id": "tomato_early_blight",
                "plant_name": "Tomato",
                "disease_name": "Early Blight",
                "scientific_name": "Alternaria solani",
                "description": "Fungal disease causing dark brown spots with concentric rings on lower leaves, leading to defoliation and reduced fruit production.",
                "symptoms": [
                    "Dark brown to black lesions on lower leaves",
                    "Concentric rings forming target-like pattern",
                    "Yellowing around spots (chlorosis)",
                    "Leaf defoliation in severe cases",
                    "Stem cankers near ground level"
                ],
                "causes": [
                    "Fungal pathogen Alternaria solani",
                    "Warm humid conditions (75-85°F)",
                    "High humidity or extended leaf wetness",
                    "Poor air circulation around plants",
                    "Infected seeds or plant debris"
                ],
                "spread_methods": [
                    "Wind-borne spores",
                    "Water splashes from rain or irrigation",
                    "Infected seeds",
                    "Contaminated tools and equipment",
                    "Plant debris in soil"
                ],
                "environmental_factors": {
                    "temperature_range": "75-85°F (24-29°C)",
                    "humidity_required": ">90% relative humidity",
                    "susceptible_stages": "Flowering to fruit development",
                    "optimal_conditions": "Warm, humid weather with poor air circulation"
                },
                "severity_levels": {
                    "mild": "<10% leaf area affected with scattered spots",
                    "moderate": "10-30% leaf area affected with coalescing lesions",
                    "severe": ">30% leaf area affected with extensive defoliation"
                },
                "reference_images": [
                    "database/reference_images/tomato_early_blight_1.jpg",
                    "database/reference_images/tomato_early_blight_2.jpg",
                    "database/reference_images/tomato_early_blight_3.jpg"
                ]
            },
            {
                "disease_id": "tomato_late_blight",
                "plant_name": "Tomato",
                "disease_name": "Late Blight",
                "scientific_name": "Phytophthora infestans",
                "description": "Devastating oomycete disease causing water-soaked lesions that rapidly expand and destroy entire plants.",
                "symptoms": [
                    "Water-soaked lesions on leaves and stems",
                    "White fungal growth on undersides of leaves in humid conditions",
                    "Dark brown to black lesions on stems",
                    "Firm, brown lesions on fruits",
                    "Rapid plant death in favorable conditions"
                ],
                "causes": [
                    "Oomycete Phytophthora infestans",
                    "Cool, wet weather (60-70°F)",
                    "High humidity and rainfall",
                    "Infected transplants or seed potatoes"
                ],
                "spread_methods": [
                    "Spores spread by wind and rain",
                    "Infected soil and plant debris",
                    "Contaminated transplants",
                    "Water movement in soil"
                ],
                "environmental_factors": {
                    "temperature_range": "60-70°F (15-21°C)",
                    "humidity_required": ">90% humidity",
                    "susceptible_stages": "All stages, especially flowering",
                    "optimal_conditions": "Cool, wet weather with extended leaf wetness"
                },
                "severity_levels": {
                    "mild": "Small, isolated water-soaked spots",
                    "moderate": "Expanding lesions with some plant tissue death",
                    "severe": "Extensive lesions leading to plant death"
                },
                "reference_images": [
                    "database/reference_images/tomato_late_blight_1.jpg",
                    "database/reference_images/tomato_late_blight_2.jpg",
                    "database/reference_images/tomato_late_blight_3.jpg"
                ]
            },
            {
                "disease_id": "tomato_septoria_leaf_spot",
                "plant_name": "Tomato",
                "disease_name": "Septoria Leaf Spot",
                "scientific_name": "Septoria lycopersici",
                "description": "Fungal disease causing small circular spots with dark borders and gray centers on lower leaves.",
                "symptoms": [
                    "Small circular spots (2-3mm) on lower leaves",
                    "Dark brown borders with grayish-white centers",
                    "Black fruiting bodies visible in spot centers",
                    "Yellowing and death of affected leaves",
                    "Progression from bottom to top of plant"
                ],
                "causes": [
                    "Fungal pathogen Septoria lycopersici",
                    "Moderate temperatures (68-77°F)",
                    "High humidity and leaf wetness",
                    "Infected plant debris"
                ],
                "spread_methods": [
                    "Rain splash dispersing spores",
                    "Wind carrying conidia",
                    "Contaminated tools",
                    "Infected transplants"
                ],
                "environmental_factors": {
                    "temperature_range": "68-77°F (20-25°C)",
                    "humidity_required": ">85% relative humidity",
                    "susceptible_stages": "Vegetative growth and fruiting",
                    "optimal_conditions": "Moderate temperatures with frequent rain"
                },
                "severity_levels": {
                    "mild": "Few spots on lower leaves",
                    "moderate": "Numerous spots causing leaf yellowing",
                    "severe": "Extensive spotting leading to severe defoliation"
                },
                "reference_images": [
                    "database/reference_images/tomato_septoria_1.jpg",
                    "database/reference_images/tomato_septoria_2.jpg",
                    "database/reference_images/tomato_septoria_3.jpg"
                ]
            }
        ]

        # POTATO DISEASES
        potato_diseases = [
            {
                "disease_id": "potato_late_blight",
                "plant_name": "Potato",
                "disease_name": "Late Blight",
                "scientific_name": "Phytophthora infestans",
                "description": "Devastating disease responsible for the Irish potato famine, causing rapid destruction of foliage and tubers.",
                "symptoms": [
                    "Water-soaked lesions on leaves and stems",
                    "White fungal growth on leaf undersides",
                    "Dark brown lesions on tubers",
                    "Rapid plant collapse in favorable conditions",
                    "Dry, firm rot in stored tubers"
                ],
                "causes": [
                    "Oomycete Phytophthora infestans",
                    "Cool, wet conditions",
                    "Infected seed potatoes",
                    "Infected plant debris"
                ],
                "spread_methods": [
                    "Sporangia spread by wind",
                    "Water splash and rain",
                    "Infected tubers",
                    "Soil-borne inoculum"
                ],
                "environmental_factors": {
                    "temperature_range": "60-70°F (15-21°C)",
                    "humidity_required": ">90% relative humidity",
                    "susceptible_stages": "All growth stages",
                    "optimal_conditions": "Cool, wet weather"
                },
                "severity_levels": {
                    "mild": "Isolated water-soaked spots",
                    "moderate": "Expanding lesions with some plant death",
                    "severe": "Complete foliage destruction and tuber infection"
                },
                "reference_images": [
                    "database/reference_images/potato_late_blight_1.jpg",
                    "database/reference_images/potato_late_blight_2.jpg",
                    "database/reference_images/potato_late_blight_3.jpg"
                ]
            },
            {
                "disease_id": "potato_early_blight",
                "plant_name": "Potato",
                "disease_name": "Early Blight",
                "scientific_name": "Alternaria solani",
                "description": "Fungal disease causing dark brown lesions with concentric rings on leaves and tubers.",
                "symptoms": [
                    "Dark brown lesions with target-like pattern on lower leaves",
                    "Leaf yellowing and premature death",
                    "Dark, sunken lesions on tubers",
                    "Reduced yield and tuber quality"
                ],
                "causes": [
                    "Fungal pathogen Alternaria solani",
                    "Warm temperatures and high humidity",
                    "Plant stress and poor nutrition"
                ],
                "spread_methods": [
                    "Conidia spread by wind and rain",
                    "Infected seed tubers",
                    "Plant debris"
                ],
                "environmental_factors": {
                    "temperature_range": "75-85°F (24-29°C)",
                    "humidity_required": ">85% relative humidity",
                    "susceptible_stages": "Maturing plants",
                    "optimal_conditions": "Warm, humid weather"
                },
                "severity_levels": {
                    "mild": "Few lesions on lower leaves",
                    "moderate": "Moderate leaf damage with some defoliation",
                    "severe": "Extensive leaf death and tuber infection"
                },
                "reference_images": [
                    "database/reference_images/potato_early_blight_1.jpg",
                    "database/reference_images/potato_early_blight_2.jpg",
                    "database/reference_images/potato_early_blight_3.jpg"
                ]
            }
        ]

        # ADD MORE CROP DISEASES HERE...
        # This is a sample - the full database would include all major crops

        # Combine all diseases
        diseases.extend(tomato_diseases)
        diseases.extend(potato_diseases)

        # Convert to Disease objects and store
        for disease_data in diseases:
            disease = Disease(**disease_data)
            self.diseases_data[disease.disease_id] = disease

        logger.info(f"Created comprehensive database with {len(self.diseases_data)} diseases")

        # Save to file
        await self.save_diseases_to_file()

    async def create_comprehensive_treatment_database(self):
        """Create comprehensive treatment database"""
        logger.info("Creating comprehensive treatment database...")

        treatments = []

        # TOMATO EARLY BLIGHT TREATMENTS
        tomato_early_blight_treatments = {
            "disease_id": "tomato_early_blight",
            "organic": {
                "priority": 1,
                "methods": [
                    {
                        "name": "Copper Fungicide",
                        "application": "Spray every 7-10 days, thoroughly covering leaf undersides",
                        "dosage": "2 tablespoons copper fungicide per gallon water",
                        "safety": "Wear gloves, avoid inhalation, wash hands after use",
                        "effectiveness": "70-80% effective when applied preventively",
                        "cost_range": "Low ($10-20 per treatment)",
                        "environmental_impact": "Copper can accumulate in soil, use sparingly"
                    },
                    {
                        "name": "Neem Oil",
                        "application": "Spray leaves thoroughly, including undersides",
                        "dosage": "2 tablespoons neem oil + 1 teaspoon gentle soap per gallon water",
                        "safety": "Non-toxic, safe for edible plants, wash produce before eating",
                        "effectiveness": "60-70% effective for mild infections",
                        "cost_range": "Very low ($5-10 per treatment)",
                        "environmental_impact": "Biodegradable, safe for beneficial insects"
                    },
                    {
                        "name": "Baking Soda Spray",
                        "application": "Apply as preventive spray every 5-7 days",
                        "dosage": "1 tablespoon baking soda + 1 teaspoon dish soap per gallon water",
                        "safety": "Very safe, food-grade ingredients",
                        "effectiveness": "50-60% effective as preventive measure",
                        "cost_range": "Very low ($2-5 per treatment)",
                        "environmental_impact": "No negative environmental impact"
                    }
                ],
                "cultural_practices": [
                    "Remove and destroy affected leaves immediately",
                    "Improve air circulation around plants through proper spacing",
                    "Water at base of plants, avoid overhead irrigation",
                    "Apply mulch to prevent soil splash onto leaves",
                    "Ensure adequate drainage and avoid waterlogged soil",
                    "Rotate crops away from tomatoes for 2-3 years"
                ],
                "effectiveness_summary": "70-80% effective when combined with cultural practices"
            },
            "chemical": {
                "priority": 2,
                "methods": [
                    {
                        "name": "Chlorothalonil",
                        "application": "Spray at first sign of disease, repeat every 7-10 days",
                        "dosage": "2 teaspoons per gallon water",
                        "safety": "Wear gloves, mask, eye protection, wash hands thoroughly",
                        "waiting_period": "0 days before harvest (check label)",
                        "effectiveness": "85-95% effective for established infections",
                        "cost_range": "Medium ($15-30 per treatment)",
                        "environmental_impact": "Broad-spectrum fungicide, can affect beneficial organisms"
                    },
                    {
                        "name": "Mancozeb",
                        "application": "Apply preventively or at first disease signs",
                        "dosage": "1-2 teaspoons per gallon water",
                        "safety": "Protective equipment required, avoid drift to sensitive areas",
                        "waiting_period": "5 days before harvest",
                        "effectiveness": "80-90% effective preventive and curative action",
                        "cost_range": "Medium ($12-25 per treatment)",
                        "environmental_impact": "Moderate environmental persistence"
                    }
                ],
                "effectiveness_summary": "85-95% effective but requires safety precautions"
            },
            "prevention": {
                "crop_rotation": "Avoid tomatoes in same soil for 3 years",
                "plant_spacing": "24-36 inches between plants for good air circulation",
                "resistant_varieties": ["Celebrity", "Mountain Fresh", "Defender", "Iron Lady"],
                "monitoring": "Check leaves twice weekly during humid periods",
                "soil_preparation": "Ensure good drainage, add organic matter",
                "water_management": "Water early morning, avoid evening watering",
                "sanitation": "Clean tools between plants, remove plant debris"
            },
            "professional_consultation": {
                "when_to_consult": [
                    "Disease affects >30% of plant foliage",
                    "Rapid spread despite treatment efforts",
                    "Yield loss exceeds 20%",
                    "Multiple diseases present simultaneously",
                    "Uncertainty about disease identification"
                ],
                "contact_resources": [
                    "Local agricultural extension office",
                    "University plant pathology clinic",
                    "Certified crop advisor",
                    "Master gardener program"
                ]
            }
        }

        treatments.append(tomato_early_blight_treatments)

        # Add more treatment entries here...

        # Convert to TreatmentInfo objects and store
        for treatment_data in treatments:
            treatment = TreatmentInfo(**treatment_data)
            self.treatments_data[treatment.disease_id] = treatment

        logger.info(f"Created comprehensive treatment database with {len(self.treatments_data)} entries")

        # Save to file
        await self.save_treatments_to_file()

    async def save_diseases_to_file(self):
        """Save diseases database to JSON file"""
        try:
            diseases_list = [disease.dict() for disease in self.diseases_data.values()]

            with open(self.diseases_file, 'w', encoding='utf-8') as f:
                json.dump(diseases_list, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(diseases_list)} diseases to {self.diseases_file}")

        except Exception as e:
            logger.error(f"Failed to save diseases to file: {str(e)}")
            raise

    async def save_treatments_to_file(self):
        """Save treatments database to JSON file"""
        try:
            treatments_list = [treatment.dict() for treatment in self.treatments_data.values()]

            with open(self.treatments_file, 'w', encoding='utf-8') as f:
                json.dump(treatments_list, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(treatments_list)} treatment entries to {self.treatments_file}")

        except Exception as e:
            logger.error(f"Failed to save treatments to file: {str(e)}")
            raise

    def build_search_indices(self):
        """Build search indices for fast disease lookup"""
        self.plant_index.clear()
        self.symptom_index.clear()
        self.cause_index.clear()

        for disease_id, disease in self.diseases_data.items():
            # Index by plant name
            plant_name = disease.plant_name.lower()
            if plant_name not in self.plant_index:
                self.plant_index[plant_name] = []
            self.plant_index[plant_name].append(disease_id)

            # Index by symptoms
            for symptom in disease.symptoms:
                symptom_key = symptom.lower().replace(" ", "_")
                if symptom_key not in self.symptom_index:
                    self.symptom_index[symptom_key] = []
                self.symptom_index[symptom_key].append(disease_id)

            # Index by causes
            for cause in disease.causes:
                cause_key = cause.lower().replace(" ", "_")
                if cause_key not in self.cause_index:
                    self.cause_index[cause_key] = []
                self.cause_index[cause_key].append(disease_id)

        logger.info(f"Built search indices: {len(self.plant_index)} plants, {len(self.symptom_index)} symptoms, {len(self.cause_index)} causes")

    async def verify_reference_images(self):
        """Verify that reference images exist and create placeholders if needed"""
        images_verified = 0
        images_missing = 0

        for disease_id, disease in self.diseases_data.items():
            disease_images = []
            for image_path in disease.reference_images:
                full_path = Path(image_path)
                if full_path.exists():
                    disease_images.append(str(full_path))
                    images_verified += 1
                else:
                    # Create placeholder image path for missing images
                    placeholder_path = self.reference_images_dir / f"{disease_id}_placeholder.jpg"
                    if not placeholder_path.exists():
                        await self.create_placeholder_image(placeholder_path, disease)
                    disease_images.append(str(placeholder_path))
                    images_missing += 1
                    logger.warning(f"Missing reference image for {disease_id}, created placeholder")

            self.reference_images_map[disease_id] = disease_images

        logger.info(f"Reference images verified: {images_verified} found, {images_missing} missing/placeholders created")

    async def create_placeholder_image(self, path: Path, disease: Disease):
        """Create a placeholder image for missing reference images"""
        try:
            from PIL import Image, ImageDraw, ImageFont

            # Create a simple placeholder image
            img = Image.new('RGB', (400, 300), color='lightgray')
            draw = ImageDraw.Draw(img)

            # Add text
            text_lines = [
                f"{disease.plant_name} - {disease.disease_name}",
                "Reference Image",
                "Placeholder"
            ]

            y_position = 50
            for line in text_lines:
                draw.text((20, y_position), line, fill='black')
                y_position += 40

            img.save(path)
            logger.debug(f"Created placeholder image: {path}")

        except Exception as e:
            logger.error(f"Failed to create placeholder image {path}: {str(e)}")

    async def get_disease_by_id(self, disease_id: str) -> Optional[Disease]:
        """Get disease information by ID"""
        return self.diseases_data.get(disease_id)

    async def get_diseases_by_plant(self, plant_name: str) -> List[Disease]:
        """Get all diseases for a specific plant"""
        plant_key = plant_name.lower()
        disease_ids = self.plant_index.get(plant_key, [])

        diseases = []
        for disease_id in disease_ids:
            disease = self.diseases_data.get(disease_id)
            if disease:
                diseases.append(disease)

        return diseases

    async def search_diseases(self, query: str, limit: int = 10) -> List[Disease]:
        """Search diseases by name, symptoms, or causes"""
        query_lower = query.lower()
        results = []
        scores = {}

        for disease_id, disease in self.diseases_data.items():
            score = 0

            # Check exact name matches
            if query_lower in disease.disease_name.lower():
                score += 10
            if query_lower in disease.plant_name.lower():
                score += 8

            # Check symptoms
            for symptom in disease.symptoms:
                if query_lower in symptom.lower():
                    score += 5
                    break

            # Check causes
            for cause in disease.causes:
                if query_lower in cause.lower():
                    score += 3
                    break

            # Check description
            if query_lower in disease.description.lower():
                score += 2

            if score > 0:
                scores[disease_id] = score
                results.append(disease)

        # Sort by score and limit results
        results.sort(key=lambda d: scores[d.disease_id], reverse=True)
        return results[:limit]

    async def get_treatment_recommendations(self, disease_id: str) -> Optional[TreatmentInfo]:
        """Get treatment recommendations for a disease"""
        return self.treatments_data.get(disease_id)

    async def get_reference_images(self, disease_id: str) -> List[str]:
        """Get reference image paths for a disease"""
        return self.reference_images_map.get(disease_id, [])

    async def get_supported_plants(self) -> List[str]:
        """Get list of all supported plants"""
        return sorted(set(disease.plant_name for disease in self.diseases_data.values()))

    async def get_disease_categories(self) -> Dict[str, List[str]]:
        """Get diseases categorized by crop type"""
        categories = {
            CropType.VEGETABLES.value: [],
            CropType.FRUITS.value: [],
            CropType.GRAINS.value: [],
            CropType.LEGUMES.value: [],
            CropType.OTHER.value: []
        }

        # Simple categorization based on plant names
        vegetable_plants = ["tomato", "potato", "pepper", "cucumber", "squash", "lettuce", "cabbage", "broccoli", "carrot", "onion"]
        fruit_plants = ["apple", "grape", "strawberry", "blueberry", "citrus", "peach", "cherry"]
        grain_plants = ["corn", "wheat", "rice", "barley", "sorghum"]
        legume_plants = ["soybean", "bean", "pea", "peanut"]

        for disease in self.diseases_data.values():
            plant_lower = disease.plant_name.lower()
            disease_id = disease.disease_id

            if any(veg in plant_lower for veg in vegetable_plants):
                categories[CropType.VEGETABLES.value].append(disease_id)
            elif any(fruit in plant_lower for fruit in fruit_plants):
                categories[CropType.FRUITS.value].append(disease_id)
            elif any(grain in plant_lower for grain in grain_plants):
                categories[CropType.GRAINS.value].append(disease_id)
            elif any(legume in plant_lower for legume in legume_plants):
                categories[CropType.LEGUMES.value].append(disease_id)
            else:
                categories[CropType.OTHER.value].append(disease_id)

        return categories

    async def get_database_stats(self) -> DatabaseStats:
        """Get comprehensive database statistics"""
        total_diseases = len(self.diseases_data)
        total_treatments = len(self.treatments_data)
        total_reference_images = sum(len(images) for images in self.reference_images_map.values())
        supported_plants = await self.get_supported_plants()

        return DatabaseStats(
            total_diseases=total_diseases,
            total_treatments=total_treatments,
            total_reference_images=total_reference_images,
            last_updated=time.time(),
            supported_plants=supported_plants
        )

    async def health_check(self) -> bool:
        """Perform health check on disease service"""
        try:
            # Check if essential data is loaded
            if not self.diseases_data:
                return False

            # Check if files are accessible
            if self.diseases_file.exists() and self.treatments_file.exists():
                return True

            return False

        except Exception as e:
            logger.error(f"Disease service health check failed: {str(e)}")
            return False

    async def cleanup(self):
        """Clean up resources"""
        logger.info("Disease service cleanup complete")

    async def add_disease(self, disease: Disease) -> bool:
        """Add a new disease to the database"""
        try:
            if disease.disease_id in self.diseases_data:
                logger.warning(f"Disease {disease.disease_id} already exists")
                return False

            self.diseases_data[disease.disease_id] = disease
            self.build_search_indices()
            await self.save_diseases_to_file()

            logger.info(f"Added disease: {disease.disease_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add disease {disease.disease_id}: {str(e)}")
            return False

    async def update_disease(self, disease_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing disease in the database"""
        try:
            if disease_id not in self.diseases_data:
                logger.warning(f"Disease {disease_id} not found")
                return False

            disease = self.diseases_data[disease_id]
            current_data = disease.dict()

            # Apply updates
            current_data.update(updates)
            updated_disease = Disease(**current_data)

            self.diseases_data[disease_id] = updated_disease
            self.build_search_indices()
            await self.save_diseases_to_file()

            logger.info(f"Updated disease: {disease_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update disease {disease_id}: {str(e)}")
            return False

    async def delete_disease(self, disease_id: str) -> bool:
        """Delete a disease from the database"""
        try:
            if disease_id not in self.diseases_data:
                logger.warning(f"Disease {disease_id} not found")
                return False

            del self.diseases_data[disease_id]

            # Remove from treatment data if exists
            if disease_id in self.treatments_data:
                del self.treatments_data[disease_id]

            self.build_search_indices()
            await self.save_diseases_to_file()
            await self.save_treatments_to_file()

            logger.info(f"Deleted disease: {disease_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete disease {disease_id}: {str(e)}")
            return False