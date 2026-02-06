# -*- coding: utf-8 -*-
"""
Core module for YOLOv12 Backend.

This module contains base interfaces, configuration management,
and custom exception classes.
"""

from src.core.interfaces import Detector, Interactor, Tracker
from src.core.config import Settings, get_settings
from src.core.exceptions import (
    ModelError,
    ModelLoadError,
    InferenceError,
    ConfigurationError,
)

__all__ = [
    "Detector",
    "Interactor",
    "Tracker",
    "Settings",
    "get_settings",
    "ModelError",
    "ModelLoadError",
    "InferenceError",
    "ConfigurationError",
]
