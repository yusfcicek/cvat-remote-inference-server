# -*- coding: utf-8 -*-
"""
Inference route handlers.

This module defines the API endpoints for model inference.

Endpoints:
    - POST /infer: Perform object detection on an image
    - GET /health: Health check endpoint
"""

import time
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from src.api.schemas.requests import ImageRequest, HealthResponse
from src.utils.image import decode_base64_image


router = APIRouter(tags=["Inference"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns the health status of the API server.

    Returns:
        Dictionary with status and timestamp.
    """
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


@router.post("/infer")
async def infer(request: ImageRequest) -> List[Dict[str, Any]]:
    """
    Perform inference on a base64-encoded image.

    This endpoint accepts a base64-encoded image and returns
    detection results in CVAT-compatible format.

    If the model is not loaded, it will be loaded automatically
    on first request (lazy initialization).

    Args:
        request: ImageRequest containing the base64 image.

    Returns:
        List of detections, each containing:
            - confidence: Detection confidence as string
            - label: Class label
            - points: Bounding box coordinates [x1, y1, x2, y2]
            - type: Shape type (always "rectangle")

    Raises:
        HTTPException: 500 if inference fails.
    """
    # Import here to avoid circular imports
    from src.api.app import get_model_instance, update_access_time

    update_access_time()

    model = get_model_instance()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model wrapper not configured. Server may still be starting."
        )

    try:
        # Decode image
        image = decode_base64_image(request.image_base64)

        # Model will auto-load if not loaded (lazy init)
        # The infer method handles lazy loading internally
        result = model.infer(image)
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
