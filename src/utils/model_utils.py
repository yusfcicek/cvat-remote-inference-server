# -*- coding: utf-8 -*-
"""
Shared utilities for model management and detection.
"""

import os
from typing import Set, Dict, Optional

IMPLEMENTATION_FILES = {
    "detector": "detector.py",
    "tracker": "tracker.py",
    "interactor": "interactor.py"
}

def detect_model_type_from_dir(model_dir: str) -> Optional[str]:
    """
    Detect model interface type by checking for implementation files.
    
    Args:
        model_dir: Absolute path to the model implementation directory.
        
    Returns:
        Type name ("detector", "tracker", "interactor") or None if not found.
    """
    for type_name, filename in IMPLEMENTATION_FILES.items():
        if os.path.exists(os.path.join(model_dir, filename)):
            return type_name
    return None

def get_implementation_path(model_dir: str, model_type: str) -> str:
    """
    Get the absolute path to the implementation file for a given type.
    """
    return os.path.join(model_dir, IMPLEMENTATION_FILES.get(model_type, "detector.py"))

def is_valid_model_dir(model_dir: str) -> bool:
    """
    Check if a directory is a valid model implementation.
    """
    return detect_model_type_from_dir(model_dir) is not None
