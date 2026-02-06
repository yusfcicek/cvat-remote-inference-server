# -*- coding: utf-8 -*-
"""
Custom exception classes for YOLOv12 Backend.

This module defines a hierarchy of custom exceptions for better
error handling and debugging throughout the application.

Exception Hierarchy:
    - ModelError (base)
        - ModelLoadError
        - InferenceError
    - ConfigurationError
    - ImageProcessingError

Example:
    >>> from src.core.exceptions import ModelLoadError
    >>> raise ModelLoadError("yolov12n", "Weights file not found")
"""


class ModelError(Exception):
    """
    Base exception for model-related errors.

    All model-specific exceptions should inherit from this class
    to allow for broad exception catching when needed.

    Attributes:
        model_name: Name of the model that caused the error.
        message: Human-readable error message.
    """

    def __init__(self, model_name: str, message: str) -> None:
        """
        Initialize ModelError.

        Args:
            model_name: The name of the model that caused the error.
            message: A descriptive error message.
        """
        self.model_name = model_name
        self.message = message
        super().__init__(f"[{model_name}] {message}")


class ModelLoadError(ModelError):
    """
    Exception raised when a model fails to load.

    This can occur due to missing weight files, incompatible
    versions, or insufficient resources.

    Example:
        >>> raise ModelLoadError("yolov12n", "yolov12n.pt not found")
    """

    pass


class InferenceError(ModelError):
    """
    Exception raised when inference fails.

    This can occur due to invalid input, model errors during
    forward pass, or post-processing failures.

    Example:
        >>> raise InferenceError("yolov12n", "Invalid image dimensions")
    """

    pass


class ConfigurationError(Exception):
    """
    Exception raised for configuration-related errors.

    This includes missing configuration files, invalid values,
    or schema validation failures.

    Attributes:
        config_key: The configuration key that caused the error.
        message: Human-readable error message.

    Example:
        >>> raise ConfigurationError("models.yolov12n.port", "Port must be > 0")
    """

    def __init__(self, config_key: str, message: str) -> None:
        """
        Initialize ConfigurationError.

        Args:
            config_key: The configuration key that caused the error.
            message: A descriptive error message.
        """
        self.config_key = config_key
        self.message = message
        super().__init__(f"Configuration error [{config_key}]: {message}")


class ImageProcessingError(Exception):
    """
    Exception raised for image processing errors.

    This includes decoding errors, format issues, or
    transformation failures.

    Attributes:
        message: Human-readable error message.
        original_error: The original exception, if any.

    Example:
        >>> raise ImageProcessingError("Invalid base64 encoding")
    """

    def __init__(
        self,
        message: str,
        original_error: Exception = None
    ) -> None:
        """
        Initialize ImageProcessingError.

        Args:
            message: A descriptive error message.
            original_error: The underlying exception, if any.
        """
        self.message = message
        self.original_error = original_error
        super().__init__(message)
