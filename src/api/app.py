# -*- coding: utf-8 -*-
"""
FastAPI application factory.

This module provides the application factory pattern for creating
FastAPI instances with proper configuration and lifecycle management.

Example:
    >>> from src.api.app import create_app
    >>> app = create_app()
    >>> # Run with: uvicorn src.api.app:app --host 0.0.0.0 --port 5001
"""

import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from src.api.routes import inference
from src.services.model_loader import LazyModelWrapper


# Global state for model management
_model_instance: Optional[LazyModelWrapper] = None
_last_access_time: float = 0.0
_idle_timeout: int = 60


def get_model_instance() -> Optional[LazyModelWrapper]:
    """
    Get the current model instance.

    Returns:
        The LazyModelWrapper instance or None if not initialized.
    """
    return _model_instance


def set_model_instance(instance: LazyModelWrapper) -> None:
    """
    Set the model instance.

    Args:
        instance: The LazyModelWrapper to use for inference.
    """
    global _model_instance
    _model_instance = instance


def update_access_time() -> None:
    """Update the last access time to current time."""
    global _last_access_time
    _last_access_time = time.time()


def set_idle_timeout(timeout: int) -> None:
    """
    Set the idle timeout for model unloading.

    Args:
        timeout: Timeout in seconds.
    """
    global _idle_timeout
    _idle_timeout = timeout


def _housekeeping_loop() -> None:
    """
    Background thread for model housekeeping.

    This function runs in a daemon thread and periodically checks
    if the model has been idle for too long. If so, it unloads
    the model to free up resources.
    """

    while True:
        time.sleep(10)  # Check every 10 seconds

        if _model_instance and _model_instance.instance is not None:
            if time.time() - _last_access_time > _idle_timeout:
                _model_instance.unload()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Handles application startup and shutdown events.
    Starts the housekeeping thread on startup.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the application.
    """
    # Startup
    housekeeping_thread = threading.Thread(
        target=_housekeeping_loop,
        daemon=True,
        name="model-housekeeping"
    )
    housekeeping_thread.start()

    yield

    # Shutdown - cleanup if needed
    pass


def create_app(
    title: str = "CVAT Custom Model API",
    version: str = "1.0.0",
    description: str = "Object Detection API with lazy model loading"
) -> FastAPI:
    """
    Create and configure a FastAPI application.

    This factory function creates a new FastAPI instance with
    all routes registered and proper lifespan management.

    Args:
        title: The API title for OpenAPI documentation.
        version: The API version.
        description: The API description.

    Returns:
        Configured FastAPI application instance.

    Example:
        >>> app = create_app()
        >>> # The app can be run with uvicorn
    """
    app = FastAPI(
        title=title,
        version=version,
        description=description,
        lifespan=lifespan
    )

    # Include routers
    app.include_router(inference.router)

    return app


# Default application instance for uvicorn
app = create_app()
