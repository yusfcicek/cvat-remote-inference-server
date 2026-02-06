# -*- coding: utf-8 -*-
"""
Tests for the orchestrator service.

These tests verify model orchestration functionality.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestModelOrchestrator:
    """Tests for ModelOrchestrator class."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly."""
        from src.services.orchestrator import ModelOrchestrator

        orchestrator = ModelOrchestrator(
            models_dir="src/models",
            poll_interval=10
        )

        assert orchestrator.models_dir == "src/models"
        assert orchestrator.poll_interval == 10
        assert orchestrator.process_registry == {}

    def test_scan_model_directories(self):
        """Test scanning for model implementation directories."""
        from src.services.orchestrator import ModelOrchestrator
        
        orchestrator = ModelOrchestrator(models_dir="src/models")
        
        with patch("os.path.exists", return_value=True):
            with patch("os.listdir", return_value=["yolov12n", "invalid_dir"]):
                with patch("os.path.isdir", return_value=True):
                    # Mock exists for implementation files
                    def special_exists(path):
                        return "yolov12n/detector.py" in path
                    
                    with patch("os.path.exists", side_effect=special_exists):
                        valid = orchestrator._scan_model_directories()
                        assert "yolov12n" in valid
                        assert "invalid_dir" not in valid

class TestLazyModelWrapper:
    """Tests for LazyModelWrapper class."""

    def test_wrapper_initialization(self):
        """Test wrapper initializes with correct state."""
        from src.services.model_loader import LazyModelWrapper

        wrapper = LazyModelWrapper("/path/to/model", "test_model", model_config={"weights": "fast.pt"})

        assert wrapper.model_dir == "/path/to/model"
        assert wrapper.model_name == "test_model"
        assert wrapper.config == {"weights": "fast.pt"}
        assert wrapper.instance is None
        assert wrapper.is_loaded is False

    def test_unload_when_not_loaded(self):
        """Test unload when model is not loaded does nothing."""
        from src.services.model_loader import LazyModelWrapper

        wrapper = LazyModelWrapper("/path/to/model", "test_model")
        wrapper.unload()  # Should not raise

        assert wrapper.instance is None


class TestConfiguration:
    """Tests for configuration loading."""

    def test_settings_loading(self):
        """Test that settings can be loaded."""
        from src.core.config import get_settings, reload_settings

        # Clear cache and reload
        settings = reload_settings()

        assert settings is not None
        assert hasattr(settings, "server")
        assert hasattr(settings, "cvat")
        assert hasattr(settings, "models")

    def test_model_config_has_config_dict(self):
        """Test that model config has the new config dictionary."""
        from src.core.config import reload_settings

        settings = reload_settings()

        if "yolov12n" in settings.models:
            model = settings.models["yolov12n"]
            assert hasattr(model, "config")
            assert isinstance(model.config, dict)
