# -*- coding: utf-8 -*-
"""
Configuration management for YOLOv12 Backend.

This module provides centralized configuration management using Pydantic
settings with YAML file support. All configuration values can be overridden
via environment variables.

Configuration Files:
    - config/models.yaml: Model configurations and server settings

Environment Variables:
    - YOLO_CONFIG_PATH: Path to models.yaml (default: config/models.yaml)
    - YOLO_SERVER_HOST: Server host (default: 0.0.0.0)
    - YOLO_CVAT_HOST: FastAPI host for CVAT/Nuclio (default: 172.21.131.139)

Example:
    >>> from src.core.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.server.host)
    '0.0.0.0'
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


# Default port range for auto-assigned models
DEFAULT_PORT_START = 5001
DEFAULT_PORT_END = 5100


class ServerConfig(BaseModel):
    """
    Server configuration settings.

    Attributes:
        host: The host address to bind the server to.
    """

    host: str = Field(default="0.0.0.0", description="Server host address")


class CVATConfig(BaseModel):
    """
    CVAT/Nuclio integration configuration.

    This configuration is used when generating Nuclio function handlers
    that call back to the FastAPI server.

    Attributes:
        fastapi_host: The IP address of the FastAPI server that Nuclio
                     functions will call for inference.
    """

    fastapi_host: str = Field(
        default="172.21.131.139",
        description="FastAPI server IP for CVAT/Nuclio integration"
    )
    nuclio_output_dir: str = Field(
        default="nuclio_functions",
        description="Directory for generated Nuclio functions"
    )


class ModelConfig(BaseModel):
    """
    Individual model configuration.

    Attributes:
        port: The port number for this model's server.
        idle_timeout_seconds: Time in seconds before unloading idle model.
        classes: List of class names for this model.
        implementation: Name of the implementation directory in src/models/.
        config: Generic configuration dictionary passed to model constructor.
    """

    port: Optional[int] = Field(default=None, description="Port number for model server")
    idle_timeout_seconds: int = Field(
        default=300,
        description="Idle timeout before model unload"
    )
    classes: Optional[list[str]] = Field(
        default=None,
        description="Class names for this model"
    )
    implementation: Optional[str] = Field(
        default=None,
        description="Name of the implementation folder in src/models"
    )
    interpreter_path: Optional[str] = Field(
        default=None,
        description="Path to the Python interpreter for this model"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generic configuration for the model instance"
    )


class Settings(BaseModel):
    """
    Application settings container.

    This class holds all configuration for the application, including
    server settings, CVAT integration settings, and per-model configurations.

    Attributes:
        server: Server configuration.
        cvat: CVAT/Nuclio integration configuration.
        models: Dictionary of model configurations keyed by model name.
    """

    server: ServerConfig = Field(default_factory=ServerConfig)
    cvat: CVATConfig = Field(default_factory=CVATConfig)
    models: Dict[str, ModelConfig] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        extra = "ignore"


def get_config_path() -> str:
    """Get the configuration file path."""
    return os.getenv("YOLO_CONFIG_PATH", "config/models.yaml")


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the parsed YAML configuration.
        Returns empty dict if file doesn't exist.

    Raises:
        yaml.YAMLError: If the YAML file is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml_config(config: Dict[str, Any], config_path: str = None) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary to save.
        config_path: Path to the YAML configuration file.
    """
    if config_path is None:
        config_path = get_config_path()

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached).

    This function loads settings from the YAML configuration file
    and caches the result for performance. The cache can be cleared
    by calling `get_settings.cache_clear()`.

    The configuration file path can be overridden using the
    YOLO_CONFIG_PATH environment variable.

    Returns:
        Settings object containing all application configuration.

    Example:
        >>> settings = get_settings()
        >>> print(settings.models.get("yolov12n"))
        ModelConfig(port=5001, idle_timeout_seconds=60, ...)
    """
    config_path = get_config_path()
    yaml_config = load_yaml_config(config_path)

    # Override with environment variables
    if os.getenv("YOLO_SERVER_HOST"):
        yaml_config.setdefault("server", {})
        yaml_config["server"]["host"] = os.getenv("YOLO_SERVER_HOST")

    if os.getenv("YOLO_CVAT_HOST"):
        yaml_config.setdefault("cvat", {})
        yaml_config["cvat"]["fastapi_host"] = os.getenv("YOLO_CVAT_HOST")

    return Settings(**yaml_config)


def reload_settings() -> Settings:
    """
    Reload settings from configuration file.

    This function clears the settings cache and reloads from disk.
    Useful when the configuration file has been modified.

    Returns:
        Fresh Settings object with updated configuration.
    """
    get_settings.cache_clear()
    return get_settings()


def get_next_available_port() -> int:
    """
    Get the next available port for a new model.

    Scans existing model configurations and returns the next
    sequential port number.

    Returns:
        Next available port number.
    """
    # Force reload from disk to get latest state
    settings = reload_settings()

    used_ports = {m.port for m in settings.models.values() if m.port is not None}

    for port in range(DEFAULT_PORT_START, DEFAULT_PORT_END):
        if port not in used_ports:
            return port

    raise RuntimeError(f"No available ports in range {DEFAULT_PORT_START}-{DEFAULT_PORT_END}")


def add_model_to_config(
    model_name: str,
    port: int,
    idle_timeout_seconds: int = 300,
    implementation: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add a new model to the configuration file.

    This function reads the current config, adds the new model,
    and writes it back to disk.

    Args:
        model_name: Name of the instance.
        port: Port number for the model server.
        idle_timeout_seconds: Idle timeout before model unload.
        implementation: Name of the implementation folder.
        config_dict: Generic configuration for the model instance.
    """
    config_path = get_config_path()
    yaml_config = load_yaml_config(config_path)

    # Ensure models dict exists
    if "models" not in yaml_config:
        yaml_config["models"] = {}

    # Check if model already exists to preserve existing settings
    if model_name in yaml_config["models"]:
        print(f"Model {model_name} already exists in config, updating...")
        existing = yaml_config["models"][model_name]
        existing["port"] = port
        existing["idle_timeout_seconds"] = idle_timeout_seconds
        if implementation:
            existing["implementation"] = implementation
        if config_dict:
            existing.setdefault("config", {}).update(config_dict)
    else:
        # Add new model configuration
        model_config_val = {
            "port": port,
            "idle_timeout_seconds": idle_timeout_seconds,
            "implementation": implementation or model_name,
            "interpreter_path": None,
            "config": config_dict or {"weights": ""}
        }

        yaml_config["models"][model_name] = model_config_val

    # Save
    save_yaml_config(yaml_config, config_path)

    # Clear cache
    get_settings.cache_clear()


def remove_model_from_config(model_name: str) -> None:
    """
    Remove a model from the configuration file.

    Args:
        model_name: Name of the model to remove.
    """
    config_path = get_config_path()
    config = load_yaml_config(config_path)

    if "models" in config and model_name in config["models"]:
        del config["models"][model_name]
        save_yaml_config(config, config_path)
        get_settings.cache_clear()
