# -*- coding: utf-8 -*-
"""
Lazy model loader implementation.

This module provides the LazyModelWrapper class which handles
on-demand model loading and unloading to optimize memory usage.

The model is only loaded when first inference is requested,
and can be automatically unloaded after a period of inactivity.
"""

import gc
import importlib
import inspect
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Type

import numpy as np

from src.core.interfaces import register_model_class
from src.core.exceptions import ModelLoadError, InferenceError
from src.utils.model_utils import detect_model_type_from_dir


class LazyModelWrapper:
    """
    Wrapper for lazy loading and unloading of AI models.

    This class manages the lifecycle of a model, loading it only
    when needed and unloading it after a period of inactivity
    to conserve memory resources.

    Attributes:
        model_dir: Directory containing the model implementation.
        model_name: Name of the model (used for logging and module import).
        config: Generic configuration dictionary passed to model constructor.
        instance: The actual model instance (None if not loaded).
        lock: Thread lock for safe concurrent access.

    Example:
        >>> wrapper = LazyModelWrapper("/path/to/models/yolov12n", "yolov12n_v1", {"weights": "v1.pt"})
        >>> results = wrapper.infer(image)  # Model loads on first call
        >>> wrapper.unload()  # Manually unload when done
    """

    def __init__(self, model_input_dir: str, model_name: str, model_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the LazyModelWrapper.

        Args:
            model_input_dir: Absolute path to the model directory.
            model_name: Name of the model (matches folder name or instance name).
            model_config: Generic configuration dictionary for the model.
        """
        self.model_dir = model_input_dir
        self.model_name = model_name
        self.config = model_config or {}
        self.instance: Optional[Any] = None
        self.lock = threading.Lock()

    def load(self) -> None:
        """
        Load the model into memory.

        This method dynamically imports the model module and instantiates
        the first valid model class found (implementing Detector, Interactor,
        or Tracker interface).

        The method is thread-safe and will only load the model once even
        if called concurrently.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        with self.lock:
            if self.instance is not None:
                return  # Already loaded

            print(f"[{self.model_name}] Loading model into memory...")
            start_time = time.time()

            try:
                # Ensure project root is in path
                project_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                # Detect implementation file type
                found_type = detect_model_type_from_dir(self.model_dir)

                if not found_type:
                    raise ModelLoadError(
                        self.model_name,
                        f"No implementation file found (detector.py, tracker.py, or interactor.py) in {self.model_dir}"
                    )

                # Construct module path
                module_name = f"src.models.{self.model_name}.{found_type}"

                # Import module
                module = importlib.import_module(module_name)

                # Find valid model class
                found_class: Optional[Type] = None
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ == module.__name__:
                        try:
                            register_model_class(obj)
                            found_class = obj
                            break
                        except TypeError:
                            continue

                if found_class is None:
                    raise ModelLoadError(
                        self.model_name,
                        f"No valid interface implementation found in {module_name}"
                    )

                # Instantiate model
                self.instance = found_class(**self.config)
                elapsed = time.time() - start_time
                print(f"[{self.model_name}] Model loaded in {elapsed:.2f}s")

            except ModelLoadError:
                raise
            except Exception as e:
                raise ModelLoadError(self.model_name, str(e)) from e

    def unload(self) -> None:
        """
        Unload the model from memory.

        This method releases the model instance and runs garbage
        collection to free up memory. Thread-safe.
        """
        with self.lock:
            if self.instance is not None:
                print(f"[{self.model_name}] Unloading model due to inactivity...")
                del self.instance
                self.instance = None

                # Release GPU memory if torch/cuda is available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except (ImportError, Exception):
                    pass

                gc.collect()
                print(f"[{self.model_name}] Model unloaded.")

    def infer(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform inference on an image.

        The model will be loaded automatically if not already loaded.

        Args:
            image: Input image as numpy array (BGR format).
            **kwargs: Additional arguments passed to the model's infer method.

        Returns:
            List of detection/inference results.

        Raises:
            InferenceError: If inference fails.
        """
        if self.instance is None:
            self.load()

        try:
            return self.instance.infer(image, **kwargs)
        except Exception as e:
            raise InferenceError(self.model_name, str(e)) from e

    @property
    def is_loaded(self) -> bool:
        """
        Check if the model is currently loaded.

        Returns:
            True if model is loaded, False otherwise.
        """
        return self.instance is not None
