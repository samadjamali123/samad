"""
Image preprocessing service for Plant Disease Detection API
"""

import logging
import numpy as np
from PIL import Image, ImageOps
import cv2
from typing import Tuple, Optional, Union
import os

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Service for preprocessing plant leaf images for disease detection
    """

    def __init__(self):
        """Initialize the image processor with default configuration"""
        self.target_size = (224, 224)  # Standard input size for most plant disease models
        self.supported_formats = ['JPEG', 'PNG', 'WEBP']
        logger.info("ImageProcessor initialized")

    def validate_image(self, image: Union[Image.Image, np.ndarray]) -> bool:
        """
        Validate that the image is suitable for processing

        Args:
            image: PIL Image or numpy array

        Returns:
            bool: True if image is valid, False otherwise
        """
        try:
            if isinstance(image, Image.Image):
                # Check if image mode is supported
                if image.mode not in ['RGB', 'RGBA', 'L']:
                    logger.warning(f"Unsupported image mode: {image.mode}")
                    return False

                # Check image dimensions
                if image.size[0] < 32 or image.size[1] < 32:
                    logger.warning(f"Image too small: {image.size}")
                    return False

                return True

            elif isinstance(image, np.ndarray):
                # Check numpy array shape
                if len(image.shape) not in [2, 3]:
                    logger.warning(f"Invalid numpy array shape: {image.shape}")
                    return False

                if len(image.shape) == 3 and image.shape[2] not in [1, 3]:
                    logger.warning(f"Invalid channel count: {image.shape[2]}")
                    return False

                return True

            else:
                logger.warning(f"Unsupported image type: {type(image)}")
                return False

        except Exception as e:
            logger.error(f"Error validating image: {str(e)}")
            return False

    def preprocess_image(self, image: Image.Image, enhance_contrast: bool = True) -> np.ndarray:
        """
        Preprocess PIL Image for model input

        Args:
            image: PIL Image to preprocess
            enhance_contrast: Whether to apply contrast enhancement

        Returns:
            np.ndarray: Preprocessed image ready for model input
        """
        try:
            if not self.validate_image(image):
                raise ValueError("Invalid image provided for preprocessing")

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.debug(f"Converted image from {image.mode} to RGB")

            # Enhance contrast if requested (helps with disease detection)
            if enhance_contrast:
                # Apply histogram equalization to improve contrast
                image = self._enhance_contrast(image)
                logger.debug("Applied contrast enhancement")

            # Resize to target dimensions
            resized_image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized image to {self.target_size}")

            # Convert to numpy array and normalize
            image_array = np.array(resized_image, dtype=np.float32)

            # Normalize pixel values to [0, 1] range
            image_array = image_array / 255.0

            logger.debug(f"Preprocessed image shape: {image_array.shape}")

            return image_array

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    def preprocess_batch(self, images: list, enhance_contrast: bool = True) -> np.ndarray:
        """
        Preprocess a batch of images

        Args:
            images: List of PIL Images
            enhance_contrast: Whether to apply contrast enhancement

        Returns:
            np.ndarray: Batch of preprocessed images
        """
        try:
            if len(images) == 0:
                raise ValueError("Empty image batch provided")

            processed_images = []

            for i, image in enumerate(images):
                try:
                    processed = self.preprocess_image(image, enhance_contrast)
                    processed_images.append(processed)
                except Exception as e:
                    logger.error(f"Error processing image {i}: {str(e)}")
                    raise ValueError(f"Failed to process image {i}: {str(e)}")

            # Stack images into batch
            batch_array = np.stack(processed_images, axis=0)
            logger.info(f"Preprocessed batch of {len(images)} images, shape: {batch_array.shape}")

            return batch_array

        except Exception as e:
            logger.error(f"Error preprocessing image batch: {str(e)}")
            raise ValueError(f"Batch preprocessing failed: {str(e)}")

    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """
        Apply contrast enhancement to help with disease detection

        Args:
            image: PIL Image

        Returns:
            PIL.Image: Enhanced image
        """
        try:
            # Convert to numpy array for OpenCV operations
            image_array = np.array(image)

            # Convert to LAB color space for better contrast enhancement
            lab_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])

            # Convert back to RGB
            enhanced_array = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

            # Convert back to PIL Image
            enhanced_image = Image.fromarray(enhanced_array.astype(np.uint8))

            return enhanced_image

        except Exception as e:
            logger.warning(f"Contrast enhancement failed, using original image: {str(e)}")
            return image

    def get_image_info(self, image: Union[Image.Image, np.ndarray]) -> dict:
        """
        Get information about an image

        Args:
            image: PIL Image or numpy array

        Returns:
            dict: Image information
        """
        try:
            if isinstance(image, Image.Image):
                return {
                    "type": "PIL.Image",
                    "mode": image.mode,
                    "size": image.size,
                    "format": image.format,
                    "has_transparency": image.mode in ['RGBA', 'LA'] or 'transparency' in image.info
                }

            elif isinstance(image, np.ndarray):
                return {
                    "type": "numpy.ndarray",
                    "shape": image.shape,
                    "dtype": str(image.dtype),
                    "min_value": float(np.min(image)),
                    "max_value": float(np.max(image)),
                    "mean_value": float(np.mean(image))
                }

            else:
                return {"type": str(type(image)), "error": "Unknown image type"}

        except Exception as e:
            logger.error(f"Error getting image info: {str(e)}")
            return {"type": "unknown", "error": str(e)}

    def resize_image(self, image: Image.Image, size: Tuple[int, int], maintain_aspect_ratio: bool = True) -> Image.Image:
        """
        Resize image to specified dimensions

        Args:
            image: PIL Image to resize
            size: Target size (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio

        Returns:
            PIL.Image: Resized image
        """
        try:
            if not self.validate_image(image):
                raise ValueError("Invalid image provided")

            if maintain_aspect_ratio:
                # Calculate new size maintaining aspect ratio
                original_width, original_height = image.size
                target_width, target_height = size

                # Calculate aspect ratios
                original_aspect = original_width / original_height
                target_aspect = target_width / target_height

                if original_aspect > target_aspect:
                    # Image is wider than target, fit to width
                    new_width = target_width
                    new_height = int(target_width / original_aspect)
                else:
                    # Image is taller than target, fit to height
                    new_height = target_height
                    new_width = int(target_height * original_aspect)

                resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                # Direct resize without maintaining aspect ratio
                resized_image = image.resize(size, Image.Resampling.LANCZOS)

            return resized_image

        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            raise ValueError(f"Image resizing failed: {str(e)}")

    def create_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (256, 256)) -> Image.Image:
        """
        Create a thumbnail version of the image

        Args:
            image: PIL Image
            size: Thumbnail size

        Returns:
            PIL.Image: Thumbnail image
        """
        try:
            # Create copy to avoid modifying original
            image_copy = image.copy()

            # Create thumbnail
            image_copy.thumbnail(size, Image.Resampling.LANCZOS)

            return image_copy

        except Exception as e:
            logger.error(f"Error creating thumbnail: {str(e)}")
            raise ValueError(f"Thumbnail creation failed: {str(e)}")