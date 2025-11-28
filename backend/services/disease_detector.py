"""
Disease detection service for Plant Disease Detection API
"""

import logging
import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DiseaseDetector:
    """
    Service for detecting plant diseases from preprocessed images
    using a trained machine learning model
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the disease detector with model configuration"""
        self.model_path = model_path or os.getenv("MODEL_PATH", "./backend/ml_models/disease_classifier.pkl")
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

        # Disease class mapping (will be loaded with model)
        self.class_names = [
            "Healthy Leaf",
            "Leaf Rust",
            "Powdery Mildew",
            "Bacterial Leaf Spot",
            "Septoria Leaf Spot",
            "Early Blight",
            "Late Blight",
            "Root Rot",
            "Mosaic Virus",
            "Aphid Damage",
            "Nutrient Deficiency",
            "Fungal Infection",
            "Pest Damage",
            "Environmental Stress",
            "Leaf Spot"
        ]

        # Scientific names for diseases
        self.scientific_names = {
            "Healthy Leaf": "N/A",
            "Leaf Rust": "Puccinia spp.",
            "Powdery Mildew": "Erysiphaceae family",
            "Bacterial Leaf Spot": "Xanthomonas spp.",
            "Septoria Leaf Spot": "Septoria spp.",
            "Early Blight": "Alternaria solani",
            "Late Blight": "Phytophthora infestans",
            "Root Rot": "Rhizoctonia solani",
            "Mosaic Virus": "Various virus species",
            "Aphid Damage": "Aphidoidea superfamily",
            "Nutrient Deficiency": "N/A",
            "Fungal Infection": "Various fungi",
            "Pest Damage": "Various arthropods",
            "Environmental Stress": "N/A",
            "Leaf Spot": "Cercospora spp."
        }

        # Severity mapping
        self.severity_mapping = {
            "Healthy Leaf": "None",
            "Leaf Rust": "Moderate",
            "Powdery Mildew": "Moderate",
            "Bacterial Leaf Spot": "High",
            "Septoria Leaf Spot": "Moderate",
            "Early Blight": "High",
            "Late Blight": "Critical",
            "Root Rot": "Critical",
            "Mosaic Virus": "High",
            "Aphid Damage": "Moderate",
            "Nutrient Deficiency": "Low",
            "Fungal Infection": "Moderate",
            "Pest Damage": "Moderate",
            "Environmental Stress": "Low",
            "Leaf Spot": "Moderate"
        }

        # Load model
        self.model = None
        self.model_loaded = False
        self._load_model()

    def _load_model(self):
        """Load the trained disease classification model"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found at {self.model_path}")
                self._create_mock_model()
                return

            # Try to load pickle model
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            if isinstance(model_data, dict) and 'model' in model_data:
                # Model saved as dictionary with metadata
                self.model = model_data['model']
                if 'class_names' in model_data:
                    self.class_names = model_data['class_names']
            else:
                # Direct model object
                self.model = model_data

            self.model_loaded = True
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Model classes: {len(self.class_names)}")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self._create_mock_model()

    def _create_mock_model(self):
        """Create a mock model for development when no trained model is available"""
        logger.warning("Creating mock model for development purposes")

        # Simple rule-based mock model
        self.model = "mock_model"
        self.model_loaded = True
        logger.info("Mock model created for development")

    def predict(self, image: np.ndarray, confidence_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Predict disease from preprocessed image

        Args:
            image: Preprocessed image array (H, W, C) or (C, H, W)
            confidence_threshold: Minimum confidence threshold (uses instance default if None)

        Returns:
            Dict containing prediction results
        """
        try:
            threshold = confidence_threshold or self.confidence_threshold

            # Validate input
            if not isinstance(image, np.ndarray):
                raise ValueError("Input must be numpy array")

            if len(image.shape) != 3:
                raise ValueError("Input must be 3D array (H, W, C)")

            # Check if model is loaded
            if not self.model_loaded:
                raise RuntimeError("Model not loaded")

            # Run prediction
            prediction_result = self._run_inference(image, threshold)

            # Add timestamp
            prediction_result['detected_at'] = datetime.utcnow().isoformat()

            logger.info(f"Prediction completed: {prediction_result['disease_name']} (confidence: {prediction_result['confidence']:.3f})")

            return prediction_result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Disease prediction failed: {str(e)}")

    def _run_inference(self, image: np.ndarray, threshold: float) -> Dict[str, Any]:
        """Run model inference on image"""
        if self.model == "mock_model":
            return self._mock_inference(image, threshold)

        try:
            # Preprocess for PyTorch model if needed
            if hasattr(self.model, 'eval'):
                return self._pytorch_inference(image, threshold)
            else:
                return self._sklearn_inference(image, threshold)

        except Exception as e:
            logger.error(f"Model inference failed: {str(e)}")
            return self._mock_inference(image, threshold)

    def _mock_inference(self, image: np.ndarray, threshold: float) -> Dict[str, Any]:
        """Mock inference for development purposes"""
        # Simulate model prediction based on simple heuristics

        # Calculate image statistics for mock logic
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)

        # Simple mock logic based on image characteristics
        if mean_intensity > 0.8:  # Very bright = likely healthy
            class_idx = 0  # Healthy Leaf
            confidence = 0.9
        elif std_intensity > 0.3:  # High variation = likely disease
            class_idx = np.random.choice([1, 2, 3, 4, 5])  # Random disease
            confidence = 0.7 + np.random.random() * 0.2
        else:  # Moderate variation
            class_idx = np.random.choice([0, 1, 2])  # Healthy or common diseases
            confidence = 0.6 + np.random.random() * 0.3

        # Ensure confidence is within valid range
        confidence = min(0.99, max(0.1, confidence))

        disease_name = self.class_names[class_idx]
        scientific_name = self.scientific_names.get(disease_name, "Unknown")
        severity = self.severity_mapping.get(disease_name, "Unknown")

        # Check confidence threshold
        if confidence < threshold:
            class_idx = 0  # Default to healthy if confidence too low
            disease_name = self.class_names[class_idx]
            scientific_name = self.scientific_names.get(disease_name, "Unknown")
            severity = self.severity_mapping.get(disease_name, "Unknown")

        return {
            "disease_name": disease_name,
            "scientific_name": scientific_name,
            "confidence": float(confidence),
            "class_index": int(class_idx),
            "severity": severity,
            "all_probabilities": [
                {"class": self.class_names[i], "probability": 0.0}
                for i in range(len(self.class_names))
            ]
        }

    def _pytorch_inference(self, image: np.ndarray, threshold: float) -> Dict[str, Any]:
        """Run PyTorch model inference"""
        try:
            # Ensure image is in correct format (C, H, W)
            if image.shape[-1] == 3:
                # (H, W, C) -> (C, H, W)
                image = np.transpose(image, (2, 0, 1))

            # Add batch dimension
            image_tensor = torch.FloatTensor(image).unsqueeze(0)

            # Set model to evaluation mode
            self.model.eval()

            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                confidence = confidence.item()
                predicted_idx = predicted.item()

                # Get all probabilities
                all_probs = probabilities.squeeze().cpu().numpy()

                # Check confidence threshold
                if confidence < threshold:
                    predicted_idx = 0  # Default to healthy

                disease_name = self.class_names[predicted_idx]
                scientific_name = self.scientific_names.get(disease_name, "Unknown")
                severity = self.severity_mapping.get(disease_name, "Unknown")

                return {
                    "disease_name": disease_name,
                    "scientific_name": scientific_name,
                    "confidence": float(confidence),
                    "class_index": int(predicted_idx),
                    "severity": severity,
                    "all_probabilities": [
                        {"class": self.class_names[i], "probability": float(all_probs[i])}
                        for i in range(len(self.class_names))
                    ]
                }

        except Exception as e:
            logger.error(f"PyTorch inference failed: {str(e)}")
            raise

    def _sklearn_inference(self, image: np.ndarray, threshold: float) -> Dict[str, Any]:
        """Run scikit-learn model inference"""
        try:
            # Flatten image for sklearn model
            image_flat = image.flatten().reshape(1, -1)

            # Get probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(image_flat)[0]
                confidence = np.max(probabilities)
                predicted_idx = np.argmax(probabilities)
            else:
                # Model doesn't support probabilities
                predicted_idx = self.model.predict(image_flat)[0]
                confidence = 0.8  # Default confidence
                probabilities = np.zeros(len(self.class_names))
                probabilities[predicted_idx] = confidence

            # Check confidence threshold
            if confidence < threshold:
                predicted_idx = 0  # Default to healthy

            disease_name = self.class_names[predicted_idx]
            scientific_name = self.scientific_names.get(disease_name, "Unknown")
            severity = self.severity_mapping.get(disease_name, "Unknown")

            return {
                "disease_name": disease_name,
                "scientific_name": scientific_name,
                "confidence": float(confidence),
                "class_index": int(predicted_idx),
                "severity": severity,
                "all_probabilities": [
                    {"class": self.class_names[i], "probability": float(probabilities[i])}
                    for i in range(len(self.class_names))
                ]
            }

        except Exception as e:
            logger.error(f"Scikit-learn inference failed: {str(e)}")
            raise

    def predict_batch(self, images: List[np.ndarray], confidence_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Predict diseases for a batch of images

        Args:
            images: List of preprocessed image arrays
            confidence_threshold: Minimum confidence threshold

        Returns:
            List of prediction results
        """
        try:
            threshold = confidence_threshold or self.confidence_threshold
            results = []

            logger.info(f"Processing batch of {len(images)} images")

            for i, image in enumerate(images):
                try:
                    result = self.predict(image, threshold)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process image {i}: {str(e)}")
                    # Add error result
                    results.append({
                        "disease_name": "Error",
                        "scientific_name": "N/A",
                        "confidence": 0.0,
                        "class_index": -1,
                        "severity": "Error",
                        "error": str(e),
                        "detected_at": datetime.utcnow().isoformat()
                    })

            successful_predictions = sum(1 for r in results if 'error' not in r)
            logger.info(f"Batch processing completed: {successful_predictions}/{len(images)} successful")

            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise RuntimeError(f"Batch disease prediction failed: {str(e)}")

    def get_class_info(self) -> Dict[str, Any]:
        """
        Get information about model classes

        Returns:
            Dict with class names, scientific names, and severity levels
        """
        return {
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "scientific_names": self.scientific_names,
            "severity_levels": list(set(self.severity_mapping.values())),
            "model_loaded": self.model_loaded,
            "model_path": self.model_path
        }

    def update_confidence_threshold(self, threshold: float):
        """
        Update the confidence threshold for predictions

        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            logger.info(f"Updated confidence threshold to {threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

    def save_model(self, path: Optional[str] = None):
        """
        Save the current model to disk

        Args:
            path: Path to save model (uses default if None)
        """
        try:
            save_path = path or self.model_path

            model_data = {
                'model': self.model,
                'class_names': self.class_names,
                'scientific_names': self.scientific_names,
                'severity_mapping': self.severity_mapping,
                'confidence_threshold': self.confidence_threshold,
                'saved_at': datetime.utcnow().isoformat()
            }

            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise RuntimeError(f"Model saving failed: {str(e)}")