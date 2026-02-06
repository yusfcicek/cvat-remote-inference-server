# -*- coding: utf-8 -*-
"""
YOLOv12 Detector Implementation.

This module provides a concrete implementation of the Detector
interface using the YOLOv12 model from Ultralytics.

The detector performs object detection on images and returns
results in CVAT-compatible format.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from src.core.interfaces import Detector
from src.core.exceptions import ModelLoadError, InferenceError


class YOLOv12Detector(Detector):
    """
    YOLOv12 object detector implementation.

    This class implements the Detector interface using YOLOv12
    from the Ultralytics library.

    Attributes:
        model: The loaded YOLO model instance.

    Example:
        >>> detector = YOLOv12Detector()
        >>> results = detector.infer(image)
        >>> for det in results:
        ...     print(f"Found {det['label']} with confidence {det['confidence']}")
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the YOLOv12 detector.

        Args:
            **kwargs: Generic configuration for the model.
                - weights: Path to specific model weights file.
        """
        weights = kwargs.get("weights")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # If weights provided, use it directly (absolute or relative to current dir)
        if weights and weights.strip():
            if os.path.isabs(weights):
                weights_path = weights
            else:
                # Try relative to implementation dir
                weights_path = os.path.join(current_dir, weights)
                
            if not os.path.exists(weights_path):
                # Try relative to project root
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                alt_weights_path = os.path.join(project_root, weights)
                if os.path.exists(alt_weights_path):
                    weights_path = alt_weights_path
        else:
            # Default weights location
            weights_dir = os.path.join(current_dir, "weights")
            weights_path = os.path.join(weights_dir, "yolov12n.pt")

        print(f"Initializing YOLOv12Detector from {current_dir}")

        try:
            if os.path.exists(weights_path):
                self.model = YOLO(weights_path)
                print(f"Loaded weights from {weights_path}")
            else:
                if weights:
                     raise ModelLoadError("yolov12n", f"Specified weights file not found: {weights_path}")
                
                # Try loading from current directory (default case)
                alt_path = os.path.join(current_dir, "yolov12n.pt")
                if os.path.exists(alt_path):
                    self.model = YOLO(alt_path)
                    print(f"Loaded weights from {alt_path}")
                else:
                    # Fallback to downloading or yolov8n
                    print("Local weights not found, attempting generic load...")
                    try:
                        self.model = YOLO("yolov12n.pt")
                    except Exception:
                        print("Warning: yolov12n.pt failed, fallback to yolov8n.pt")
                        self.model = YOLO("yolov8n.pt")

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError("yolov12n", str(e)) from e

    def infer(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.25,
        device: str = "cpu",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform object detection on an image.

        Args:
            image: Input image as numpy array in BGR format.
            confidence_threshold: Minimum confidence for detections.
            device: Device to run inference on ("cpu", "cuda:0", etc.).
            **kwargs: Additional arguments passed to YOLO model.

        Returns:
            List of detection dictionaries, each containing:
                - confidence: Detection confidence as string
                - label: Class label name
                - points: Bounding box [x1, y1, x2, y2]
                - type: Always "rectangle"

        Raises:
            InferenceError: If detection fails.

        Example:
            >>> results = detector.infer(image, confidence_threshold=0.5)
            >>> print(f"Found {len(results)} objects")
        """
        try:
            # Run YOLO inference
            results = self.model(
                image,
                device=device,
                verbose=False,
                conf=confidence_threshold,
                **kwargs
            )

            detections = []

            for result in results:
                boxes = result.boxes

                for box in boxes:
                    # Extract bounding box coordinates
                    coords = box.xyxy[0].tolist()

                    # Extract confidence
                    conf = box.conf[0].item()

                    # Extract class information
                    cls_id = int(box.cls[0].item())
                    label = result.names[cls_id]

                    # Create CVAT-compatible detection
                    detection = {
                        "confidence": str(float(conf)),
                        "label": label,
                        "points": coords,
                        "type": "rectangle"
                    }
                    detections.append(detection)

            return detections

        except Exception as e:
            raise InferenceError("yolov12n", str(e)) from e
