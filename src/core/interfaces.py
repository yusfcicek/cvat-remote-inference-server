# -*- coding: utf-8 -*-
"""
Base interfaces for AI model implementations.

This module defines abstract base classes that all model implementations
must inherit from. Each model should implement exactly one interface.

Interfaces:
    - Detector: For object detection models (YOLO, R-CNN, etc.)
    - Interactor: For interactive AI models (smart polygon, etc.)
    - Tracker: For object tracking models

Example:
    >>> from src.core.interfaces import Detector
    >>> class MyDetector(Detector):
    ...     def infer(self, image, **kwargs):
    ...         # Implementation here
    ...         return [{"label": "person", "confidence": "0.95", ...}]
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

import numpy as np


class Interactor(ABC):
    """
    Abstract base class for AI interactors.

    Interactors are models that provide interactive AI capabilities
    such as smart polygon implementations, segmentation with user input, etc.

    All subclasses must implement the `infer` method.

    Attributes:
        None

    Example:
        >>> class SmartPolygon(Interactor):
        ...     def infer(self, image, points=None, **kwargs):
        ...         # Process image with user-provided points
        ...         return [{"points": [...], "type": "polygon"}]
    """

    @abstractmethod
    def infer(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform inference on the given image.

        Args:
            image: Input image as numpy array in BGR format (OpenCV).
            **kwargs: Additional arguments specific to the implementation.

        Returns:
            List of dictionaries containing inference results.
            Each dictionary should contain at minimum:
                - label: str - The class label
                - confidence: str - Confidence score as string
                - points: List[float] - Coordinates
                - type: str - Shape type (polygon, mask, etc.)

        Raises:
            InferenceError: If inference fails.
        """
        pass


class Detector(ABC):
    """
    Abstract base class for AI detectors.

    Detectors are models that perform object detection on images,
    returning bounding boxes or other geometric shapes.

    All subclasses must implement the `infer` method.

    Example:
        >>> class YOLODetector(Detector):
        ...     def infer(self, image, **kwargs):
        ...         # Run YOLO detection
        ...         return [{"label": "car", "confidence": "0.89", ...}]
    """

    @abstractmethod
    def infer(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform object detection on the given image.

        Args:
            image: Input image as numpy array in BGR format (OpenCV).
            **kwargs: Additional arguments (e.g., confidence threshold).

        Returns:
            List of dictionaries containing detection results.
            Each dictionary should contain:
                - label: str - The detected class label
                - confidence: str - Confidence score as string
                - points: List[float] - Bounding box as [x1, y1, x2, y2]
                - type: str - Always "rectangle" for detectors

        Raises:
            InferenceError: If detection fails.
        """
        pass


class Tracker(ABC):
    """
    Abstract base class for AI trackers.

    Trackers are models that track objects across video frames,
    maintaining object identity over time.

    All subclasses must implement the `infer` method.

    Example:
        >>> class DeepSORTTracker(Tracker):
        ...     def infer(self, image, **kwargs):
        ...         # Track objects in frame
        ...         return [{"label": "person", "track_id": 1, ...}]
    """

    @abstractmethod
    def infer(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform object tracking on the given image/frame.

        Args:
            image: Input image/frame as numpy array in BGR format.
            **kwargs: Additional arguments (e.g., previous detections).

        Returns:
            List of dictionaries containing tracking results.
            Each dictionary should contain:
                - label: str - The tracked class label
                - confidence: str - Confidence score as string
                - points: List[float] - Bounding box coordinates
                - type: str - Shape type
                - track_id: int - Unique identifier for tracked object

        Raises:
            InferenceError: If tracking fails.
        """
        pass


def register_model_class(cls: Type) -> None:
    """
    Validate and register a model class.

    This function ensures that a class implements exactly one interface
    from the set {Interactor, Detector, Tracker}. It is called during
    dynamic model loading to validate model implementations.

    Args:
        cls: The class to validate and register.

    Raises:
        TypeError: If the class implements zero or multiple interfaces.

    Example:
        >>> class MyDetector(Detector):
        ...     def infer(self, image, **kwargs):
        ...         return []
        >>> register_model_class(MyDetector)  # OK
        >>>
        >>> class BadModel(Detector, Tracker):  # Multiple interfaces
        ...     def infer(self, image, **kwargs):
        ...         return []
        >>> register_model_class(BadModel)  # Raises TypeError
    """
    interfaces = {Interactor, Detector, Tracker}
    implemented = {iface for iface in interfaces if issubclass(cls, iface)}

    if len(implemented) > 1:
        raise TypeError(
            f"Class {cls.__name__} implements multiple interfaces: "
            f"{[i.__name__ for i in implemented]}. "
            "It must implement exactly one."
        )
    if len(implemented) == 0:
        raise TypeError(
            f"Class {cls.__name__} must implement one of "
            f"{Interactor.__name__}, {Detector.__name__}, "
            f"or {Tracker.__name__}."
        )
