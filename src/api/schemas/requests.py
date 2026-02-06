# -*- coding: utf-8 -*-
"""
Pydantic request and response schemas.

This module defines the data models used for API request
validation and response serialization.
"""

from pydantic import BaseModel, Field


class ImageRequest(BaseModel):
    """
    Request model for inference endpoint.

    Attributes:
        image_base64: Base64-encoded image string.
                     Can optionally include data URI prefix.
    """

    image_base64: str = Field(
        ...,
        description="Base64-encoded image, optionally with data URI prefix"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.

    Attributes:
        status: Current health status.
        timestamp: Unix timestamp of the response.
    """

    status: str = Field(..., description="Health status")
    timestamp: float = Field(..., description="Response timestamp")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": 1706367600.0
            }
        }


class DetectionResult(BaseModel):
    """
    Model for a single detection result.

    Attributes:
        confidence: Detection confidence score as string.
        label: Detected class label.
        points: Bounding box coordinates [x1, y1, x2, y2].
        type: Shape type (rectangle, polygon, etc.).
    """

    confidence: str = Field(..., description="Confidence score as string")
    label: str = Field(..., description="Class label")
    points: list = Field(..., description="Bounding box coordinates")
    type: str = Field(default="rectangle", description="Shape type")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "confidence": "0.95",
                "label": "person",
                "points": [100.0, 150.0, 300.0, 450.0],
                "type": "rectangle"
            }
        }
