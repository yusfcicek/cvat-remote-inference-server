# -*- coding: utf-8 -*-
"""
Tests for API endpoints.

These tests verify the functionality of the inference API
and health check endpoints.
"""

import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check_returns_healthy(self):
        """Test that health endpoint returns healthy status."""
        from fastapi.testclient import TestClient
        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestInferenceEndpoint:
    """Tests for the inference endpoint."""

    def test_infer_without_model_returns_503(self):
        """Test that inference without model returns service unavailable."""
        from fastapi.testclient import TestClient
        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        # Create simple test image
        test_image = base64.b64encode(b"fake_image_data").decode()

        response = client.post(
            "/infer",
            json={"image_base64": test_image}
        )

        # Should return 503 since model is not initialized
        assert response.status_code == 503

    def test_infer_with_mocked_model(self):
        """Test inference with a mocked model."""
        from fastapi.testclient import TestClient
        from src.api import app as app_module
        from src.api.app import create_app

        # Create mock model
        mock_model = MagicMock()
        mock_model.infer.return_value = [
            {
                "confidence": "0.95",
                "label": "person",
                "points": [100.0, 100.0, 200.0, 200.0],
                "type": "rectangle"
            }
        ]

        # Patch the model getter
        with patch.object(app_module, "get_model_instance", return_value=mock_model):
            with patch("src.api.routes.inference.decode_base64_image") as mock_decode:
                mock_decode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

                app = create_app()
                client = TestClient(app)

                test_image = base64.b64encode(b"fake_image_data").decode()
                response = client.post(
                    "/infer",
                    json={"image_base64": test_image}
                )

                assert response.status_code == 200
                data = response.json()
                assert len(data) == 1
                assert data[0]["label"] == "person"


class TestImageUtils:
    """Tests for image utility functions."""

    def test_decode_valid_base64_image(self):
        """Test decoding a valid base64 image."""
        import cv2
        from src.utils.image import decode_base64_image

        # Create a simple test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[25:75, 25:75] = [255, 255, 255]

        # Encode to base64
        _, buffer = cv2.imencode('.jpg', test_img)
        b64_str = base64.b64encode(buffer).decode()

        # Decode and verify
        result = decode_base64_image(b64_str)

        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)

    def test_decode_with_data_uri_prefix(self):
        """Test decoding base64 with data URI prefix."""
        import cv2
        from src.utils.image import decode_base64_image

        test_img = np.zeros((50, 50, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', test_img)
        b64_str = base64.b64encode(buffer).decode()

        # Add data URI prefix
        data_uri = f"data:image/jpeg;base64,{b64_str}"

        result = decode_base64_image(data_uri)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 50
        assert result.shape[1] == 50

    def test_decode_invalid_base64_raises_error(self):
        """Test that invalid base64 raises ImageProcessingError."""
        from src.utils.image import decode_base64_image
        from src.core.exceptions import ImageProcessingError

        with pytest.raises(ImageProcessingError):
            decode_base64_image("not_valid_base64!!!")
