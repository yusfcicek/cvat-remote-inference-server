# -*- coding: utf-8 -*-
"""
Image processing utilities.

This module provides utility functions for image processing,
including base64 decoding and format conversion.
"""

import base64

import cv2
import numpy as np

from src.core.exceptions import ImageProcessingError


def decode_base64_image(data: str) -> np.ndarray:
    """
    Decode a base64-encoded image string into a numpy array.

    This function handles both raw base64 strings and data URI
    format (e.g., "data:image/jpeg;base64,/9j/4AAQ...").

    Args:
        data: Base64-encoded image string, optionally with
              data URI prefix.

    Returns:
        Decoded image as numpy array in BGR format (OpenCV).

    Raises:
        ImageProcessingError: If decoding fails.

    Example:
        >>> import base64
        >>> with open("image.jpg", "rb") as f:
        ...     b64 = base64.b64encode(f.read()).decode()
        >>> image = decode_base64_image(b64)
        >>> print(image.shape)
        (480, 640, 3)
    """
    try:
        # Strip data URI prefix if present
        if "," in data:
            header, encoded = data.split(",", 1)
        else:
            encoded = data

        # Decode base64
        decoded_bytes = base64.b64decode(encoded)

        # Convert to numpy array
        np_arr = np.frombuffer(decoded_bytes, np.uint8)

        # Decode image using OpenCV
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image bytes")

        return image

    except Exception as e:
        raise ImageProcessingError(
            f"Invalid base64 image data: {str(e)}",
            original_error=e
        )


def encode_image_to_base64(
    image: np.ndarray,
    format: str = "jpeg",
    quality: int = 95
) -> str:
    """
    Encode a numpy array image to base64 string.

    Args:
        image: Image as numpy array in BGR format.
        format: Output format ("jpeg", "png").
        quality: JPEG quality (1-100), ignored for PNG.

    Returns:
        Base64-encoded image string.

    Raises:
        ImageProcessingError: If encoding fails.

    Example:
        >>> b64 = encode_image_to_base64(image, format="jpeg", quality=90)
        >>> print(b64[:20])
        '/9j/4AAQSkZJRgABAQ...'
    """
    try:
        format = format.lower()

        if format == "jpeg":
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success, buffer = cv2.imencode(".jpg", image, encode_param)
        elif format == "png":
            success, buffer = cv2.imencode(".png", image)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if not success:
            raise ValueError("Failed to encode image")

        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        raise ImageProcessingError(
            f"Failed to encode image: {str(e)}",
            original_error=e
        )


def resize_image(
    image: np.ndarray,
    max_size: int = 1280,
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize an image while optionally maintaining aspect ratio.

    Args:
        image: Input image as numpy array.
        max_size: Maximum dimension (width or height).
        keep_aspect_ratio: If True, maintain aspect ratio.

    Returns:
        Resized image as numpy array.

    Example:
        >>> resized = resize_image(image, max_size=640)
        >>> print(resized.shape)
    """
    h, w = image.shape[:2]

    if keep_aspect_ratio:
        if max(h, w) <= max_size:
            return image

        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
    else:
        new_w = new_h = max_size

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
