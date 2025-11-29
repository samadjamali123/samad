"""
Custom CNN Model for Plant Disease Detection

This module implements a deep learning model for plant disease detection and symptom analysis
using ResNet-50 backbone with disease-specific fine-tuning. The model supports multi-task
learning for disease classification and severity estimation with spatial attention maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class DiseaseDetector(nn.Module):
    """
    Custom CNN model for plant disease detection using ResNet-50 backbone.
    
    Features:
    - ResNet-50 backbone pre-trained on ImageNet
    - Disease-specific fine-tuning for agricultural applications
    - Multi-task learning (disease classification + severity estimation)
    - Spatial attention maps for symptom localization
    - Support for 20+ common diseases per crop category
    - Confidence calibration using temperature scaling
    - Grad-CAM integration for interpretability
    """
    
    # Disease categories by crop type
    DISEASE_CATEGORIES = {
        'tomato': [
            'early_blight', 'late_blight', 'septoria_leaf_spot', 'target_spot',
            'bacterial_spot', 'powdery_mildew', 'leaf_mold', 'mosaic_virus',
            'yellow_leaf_curl_virus', 'two_spotted_spider_mite', 'healthy'
        ],
        'potato': [
            'early_blight', 'late_blight', 'healthy'
        ],
        'apple': [
            'apple_scab', 'black_rot', 'cedar_apple_rust', 'fire_blight', 'healthy'
        ],
        'grape': [
            'black_rot', 'esca_black_measles', 'leaf_blight_isariopsis_leaf_spot',
            'healthy'
        ],
        'corn': [
            'common_rust', 'gray_leaf_spot', 'blight', 'healthy'
        ],
        'citrus': [
            'black_rot', 'canker', 'canker_greening', 'healthy'
        ],
        'pepper': [
            'bacterial_spot', 'healthy'
        ],
        'strawberry': [
            'leaf_scorch', 'healthy'
        ]
    }
    
    # Severity levels
    SEVERITY_LEVELS = {
        0: 'healthy',
        1: 'very_mild',      # 1-10% affected
        2: 'mild',           # 11-25% affected  
        3: 'moderate',       # 26-50% affected
        4: 'severe',         # 51-75% affected
        5: 'very_severe'     # 76-100% affected
    }
    
    # Global disease mapping (simplified)
    ALL_DISEASES = []
    for crop_diseases in DISEASE_CATEGORIES.values():
        ALL_DISEASES.extend(crop_diseases)
    DISEASE_TO_IDX = {disease: idx for idx, disease in enumerate(ALL_DISEASES)}
    
    def __init__(
        self,
        num_classes: int = len(ALL_DISEASES),
        num_severity_levels: int = len(SEVERITY_LEVELS),
        pretrained: bool = True,
        dropout_rate: float = 0.4,
        attention_dim: int = 256,
        use_spatial_attention: bool = True
    ):
        """
        Initialize the disease detector model.
        
        Args:
            num_classes: Number of disease classes to predict
            num_severity_levels: Number of severity levels
            pretrained: Whether to use pre-trained weights
            dropout_rate: Dropout rate for regularization
            attention_dim: Dimension for attention mechanism
            use_spatial_attention: Whether to use spatial attention
        """
        super(DiseaseDetector, self).__init__()
        
        self.num_classes = num_classes
        self.num_severity_levels = num_severity_levels
        self.attention_dim = attention_dim
        self.use_spatial_attention = use_spatial_attention
        
        # Load ResNet-50 backbone
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Remove the original classifier
        self.backbone.fc = nn.Identity()
        
        # Get feature dimension from ResNet-50
        backbone_features = 2048  # ResNet-50 output features
        
        # Spatial attention mechanism for symptom localization
        if use_spatial_attention:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(backbone_features, attention_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(attention_dim, 1, kernel_size=1),
                nn.Sigmoid()
            )
        
        # Disease classification branch
        self.disease_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Severity estimation branch
        self.severity_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_severity_levels)
        )
        
        # Feature fusion for multi-task learning
        self.feature_fusion = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate)
        )
        
        # Temperature scaling for confidence calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for new layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Dictionary containing:
            - disease_logits: Raw disease classification scores
            - disease_probs: Calibrated disease probabilities
            - severity_logits: Raw severity scores
            - severity_probs: Severity probabilities
            - attention_map: Spatial attention map for localization
            - features: Fused feature representations
        """
        # Extract features using ResNet backbone
        features = self.backbone(x)  # Shape: (batch_size, 2048, 7, 7)
        
        # Spatial attention for symptom localization
        if self.use_spatial_attention:
            attention_map = self.spatial_attention(features)  # Shape: (batch_size, 1, 7, 7)
            attended_features = features * attention_map
        else:
            attention_map = torch.ones(features.shape[0], 1, features.shape[2], features.shape[3], 
                                     device=features.device)
            attended_features = features
        
        # Disease classification
        disease_logits = self.disease_classifier(attended_features)
        
        # Apply temperature scaling for calibration
        disease_logits_calibrated = disease_logits / self.temperature
        disease_probs = F.softmax(disease_logits_calibrated, dim=1)
        
        # Severity estimation
        severity_logits = self.severity_estimator(attended_features)
        severity_probs = F.softmax(severity_logits, dim=1)
        
        # Feature fusion
        global_features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        fused_features = self.feature_fusion(global_features)
        
        return {
            'disease_logits': disease_logits,
            'disease_probs': disease_probs,
            'severity_logits': severity_logits,
            'severity_probs': severity_probs,
            'attention_map': attention_map,
            'features': fused_features,
            'raw_features': global_features,
            'temperature': self.temperature
        }
    
    def predict_disease(
        self,
        x: torch.Tensor,
        top_k: int = 5,
        confidence_threshold: float = 0.3
    ) -> Dict[str, torch.Tensor]:
        """
        Predict disease with confidence scores and severity.
        
        Args:
            x: Input tensor
            top_k: Number of top predictions to return
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        with torch.no_grad():
            outputs = self.forward(x)
            disease_probs = outputs['disease_probs']
            severity_probs = outputs['severity_probs']
            
            # Get top-k disease predictions
            top_disease_probs, top_disease_indices = torch.topk(disease_probs, top_k, dim=1)
            
            # Convert to disease names
            top_diseases = [[self.ALL_DISEASES[idx.item()] for idx in indices] 
                           for indices in top_disease_indices]
            
            # Get severity predictions (highest probability)
            top_severity_probs, top_severity_indices = torch.max(severity_probs, dim=1)
            
            # Filter by confidence threshold
            batch_results = []
            for batch_idx in range(x.shape[0]):
                filtered_predictions = []
                batch_top_probs = top_disease_probs[batch_idx]
                batch_top_diseases = top_diseases[batch_idx]
                
                for disease, prob in zip(batch_top_diseases, batch_top_probs):
                    if prob.item() >= confidence_threshold and disease != 'healthy':
                        filtered_predictions.append({
                            'disease': disease,
                            'confidence': prob.item(),
                            'severity': self.SEVERITY_LEVELS[top_severity_indices[batch_idx].item()],
                            'severity_confidence': top_severity_probs[batch_idx].item()
                        })
                
                # If no diseases meet threshold, check if healthy is likely
                healthy_idx = self.DISEASE_TO_IDX.get('healthy', 0)
                healthy_confidence = disease_probs[batch_idx, healthy_idx].item()
                
                if not filtered_predictions and healthy_confidence > 0.5:
                    filtered_predictions.append({
                        'disease': 'healthy',
                        'confidence': healthy_confidence,
                        'severity': 'healthy',
                        'severity_confidence': 1.0
                    })
                elif not filtered_predictions and batch_top_probs[0] > 0:
                    # Return highest confidence disease even if below threshold
                    filtered_predictions.append({
                        'disease': batch_top_diseases[0],
                        'confidence': batch_top_probs[0].item(),
                        'severity': self.SEVERITY_LEVELS[top_severity_indices[batch_idx].item()],
                        'severity_confidence': top_severity_probs[batch_idx].item()
                    })
                
                batch_results.append(filtered_predictions)
            
            return {
                'predictions': batch_results,
                'top_k_diseases': top_diseases,
                'top_disease_probabilities': top_disease_probs.cpu().numpy(),
                'severity_predictions': [self.SEVERITY_LEVELS[idx.item()] 
                                      for idx in top_severity_indices],
                'severity_probabilities': top_severity_probs.cpu().numpy(),
                'attention_maps': outputs['attention_map'].cpu().numpy(),
                'temperature': outputs['temperature'].item()
            }
    
    def get_grad_cam(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        target_layer: str = 'layer4'
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for interpretability.
        
        Args:
            x: Input tensor
            target_class: Target class for Grad-CAM (if None, use predicted class)
            target_layer: Layer to use for Grad-CAM
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        self.eval()
        x.requires_grad_()
        
        # Forward pass
        outputs = self.forward(x)
        disease_probs = outputs['disease_probs']
        
        # Use predicted class if target_class not provided
        if target_class is None:
            target_class = torch.argmax(disease_probs, dim=1)
        
        # Get features from target layer
        features = None
        for name, module in self.backbone.named_modules():
            if name == target_layer:
                features = module
                break
        
        if features is None:
            # Fallback to last conv layer
            features = list(self.backbone.children())[-2]
        
        # Backward pass
        self.zero_grad()
        loss = disease_probs.gather(1, target_class.unsqueeze(1)).sum()
        loss.backward()
        
        # Get gradients
        gradients = x.grad
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of forward activation maps
        grad_cam = torch.sum(features * weights, dim=1)
        grad_cam = F.relu(grad_cam)
        
        # Normalize and resize
        grad_cam = grad_cam - grad_cam.min()
        grad_cam = grad_cam / grad_cam.max()
        
        # Resize to input size
        grad_cam = F.interpolate(
            grad_cam.unsqueeze(1), 
            size=(x.shape[2], x.shape[3]), 
            mode='bilinear', 
            align_corners=False
        ).squeeze().cpu().numpy()
        
        return grad_cam
    
    @classmethod
    def get_preprocessing_transforms(cls) -> transforms.Compose:
        """
        Get preprocessing transforms for ResNet-50.
        
        Returns:
            Compose of preprocessing transforms
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet-50 input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]    # ImageNet stds
            )
        ])
    
    @classmethod
    def get_data_augmentation_transforms(cls) -> transforms.Compose:
        """
        Get data augmentation transforms for training.
        
        Returns:
            Compose of augmentation transforms
        """
        return transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def calibrate_temperature(self, validation_loader, device='cpu'):
        """
        Calibrate temperature scaling on validation set.
        
        Args:
            validation_loader: Validation data loader
            device: Device to use for calibration
        """
        self.to(device)
        self.eval()
        
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_model():
            self.eval()
            all_logits = []
            all_labels = []
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(validation_loader):
                    data, target = data.to(device), target.to(device)
                    outputs = self.forward(data)
                    logits = outputs['disease_logits']
                    all_logits.append(logits)
                    all_labels.append(target)
            
            all_logits = torch.cat(all_logits)
            all_labels = torch.cat(all_labels)
            
            loss = nll_criterion(all_logits / self.temperature, all_labels)
            return loss
        
        optimizer.step(eval_model)
        
        logger.info(f"Temperature calibrated to: {self.temperature.item():.4f}")
    
    def save_model(self, path: str):
        """
        Save model state dictionary.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'num_severity_levels': self.num_severity_levels,
            'attention_dim': self.attention_dim,
            'use_spatial_attention': self.use_spatial_attention,
            'disease_categories': self.DISEASE_CATEGORIES,
            'severity_levels': self.SEVERITY_LEVELS
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cpu') -> 'DiseaseDetector':
        """
        Load model from checkpoint.
        
        Args:
            path: Path to the model checkpoint
            device: Device to load the model on
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            num_classes=checkpoint['num_classes'],
            num_severity_levels=checkpoint['num_severity_levels'],
            attention_dim=checkpoint['attention_dim'],
            use_spatial_attention=checkpoint['use_spatial_attention']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded from {path}")
        return model
    
    def export_to_onnx(self, path: str, input_shape: Tuple[int, ...] = (1, 3, 224, 224)):
        """
        Export model to ONNX format for cross-platform compatibility.
        
        Args:
            path: Path to save ONNX model
            input_shape: Input tensor shape
        """
        self.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            self,
            dummy_input,
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['disease_output', 'severity_output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'disease_output': {0: 'batch_size'},
                'severity_output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to ONNX format: {path}")
    
    def get_model_info(self) -> Dict:
        """
        Get model information and statistics.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'DiseaseDetector',
            'backbone': 'ResNet-50',
            'num_disease_classes': self.num_classes,
            'num_severity_levels': self.num_severity_levels,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'supported_diseases': self.ALL_DISEASES,
            'supported_crops': list(self.DISEASE_CATEGORIES.keys()),
            'input_size': (224, 224),
            'pretrained': True,
            'spatial_attention': self.use_spatial_attention,
            'multi_task_learning': True,
            'confidence_calibration': True,
            'grad_cam_support': True
        }


class DiseaseDetectorInference:
    """
    Inference wrapper for DiseaseDetector with preprocessing and postprocessing.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[DiseaseDetector] = None,
        device: str = 'cpu',
        confidence_threshold: float = 0.3
    ):
        """
        Initialize inference wrapper.
        
        Args:
            model_path: Path to saved model
            model: Pre-loaded model instance
            device: Device for inference
            confidence_threshold: Minimum confidence threshold
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = DiseaseDetector.load_model(model_path, device)
        else:
            raise ValueError("Either model_path or model must be provided")
        
        self.model.eval()
        self.preprocess = DiseaseDetector.get_preprocessing_transforms()
    
    def predict_image(
        self,
        image,
        generate_grad_cam: bool = False
    ) -> Dict[str, any]:
        """
        Predict disease from a single image.
        
        Args:
            image: PIL Image or numpy array
            generate_grad_cam: Whether to generate Grad-CAM visualization
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        if hasattr(image, 'convert'):  # PIL Image
            image_tensor = self.preprocess(image).unsqueeze(0)
        else:  # Numpy array
            from torchvision.transforms import ToPILImage
            pil_image = ToPILImage()(image)
            image_tensor = self.preprocess(pil_image).unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model.predict_disease(
                image_tensor, 
                confidence_threshold=self.confidence_threshold
            )
            
            batch_result = predictions['predictions'][0]
            
            # Generate Grad-CAM if requested
            grad_cam = None
            if generate_grad_cam and batch_result and batch_result[0]['disease'] != 'healthy':
                disease_name = batch_result[0]['disease']
                if disease_name in DiseaseDetector.DISEASE_TO_IDX:
                    target_class = DiseaseDetector.DISEASE_TO_IDX[disease_name]
                    grad_cam = self.model.get_grad_cam(image_tensor, target_class)
            
            return {
                'detections': batch_result,
                'primary_disease': batch_result[0] if batch_result else None,
                'is_healthy': batch_result and batch_result[0]['disease'] == 'healthy',
                'grad_cam': grad_cam,
                'attention_map': predictions['attention_maps'][0] if len(predictions['attention_maps']) > 0 else None,
                'model_confidence': batch_result[0]['confidence'] if batch_result else 0.0,
                'temperature': predictions['temperature'],
                'num_detections': len(batch_result)
            }
    
    def get_disease_info(self, disease_name: str) -> Dict:
        """
        Get information about a specific disease.
        
        Args:
            disease_name: Name of the disease
            
        Returns:
            Dictionary with disease information
        """
        # This would be expanded with comprehensive disease database
        disease_info = {
            'early_blight': {
                'scientific_name': 'Alternaria solani',
                'causal_organism': 'Fungus',
                'symptoms': ['Small dark spots', 'Concentric rings', 'Yellowing'],
                'favorable_conditions': ['High humidity', 'Moderate temperatures'],
                'spread_methods': ['Wind', 'Water splash', 'Infected seeds'],
                'treatment_difficulty': 'moderate'
            },
            'late_blight': {
                'scientific_name': 'Phytophthora infestans',
                'causal_organism': 'Oomycete',
                'symptoms': ['Water-soaked lesions', 'White fungal growth', 'Rapid spreading'],
                'favorable_conditions': ['Cool wet weather', 'High humidity'],
                'spread_methods': ['Spores in air', 'Water splash'],
                'treatment_difficulty': 'severe'
            },
            'powdery_mildew': {
                'scientific_name': 'Oidium spp.',
                'causal_organism': 'Fungus',
                'symptoms': ['White powdery coating', 'Leaf distortion', 'Reduced photosynthesis'],
                'favorable_conditions': ['Dry conditions', 'Moderate temperatures'],
                'spread_methods': ['Wind-borne spores'],
                'treatment_difficulty': 'mild'
            }
        }
        
        return disease_info.get(disease_name.lower(), {
            'scientific_name': f'{disease_name.title()} pathogen',
            'causal_organism': 'Unknown',
            'symptoms': ['Visual symptoms detected'],
            'favorable_conditions': ['Environmental factors'],
            'spread_methods': ['Various transmission methods'],
            'treatment_difficulty': 'unknown'
        })


# Factory function for easy model creation
def create_disease_detector(
    pretrained: bool = True,
    device: str = 'cpu',
    confidence_threshold: float = 0.3
) -> DiseaseDetectorInference:
    """
    Factory function to create a disease detector for inference.
    
    Args:
        pretrained: Whether to use pre-trained weights
        device: Device for inference
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        DiseaseDetectorInference instance
    """
    model = DiseaseDetector(pretrained=pretrained)
    model.to(device)
    model.eval()
    
    return DiseaseDetectorInference(
        model=model,
        device=device,
        confidence_threshold=confidence_threshold
    )