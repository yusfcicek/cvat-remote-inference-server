# -*- coding: utf-8 -*-
"""
Main entry point for CVAT Custom Model Orchestrator.

This script starts the model orchestrator which manages
all configured model servers.

Usage:
    python main.py

The orchestrator will:
    - Read configuration from config/yolo_config.yaml
    - Start model servers for each configured model
    - Monitor and restart crashed servers
    - Stop servers for removed models
"""

import os
import sys

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.services.orchestrator import ModelOrchestrator


def main() -> None:
    """
    Main entry point.

    Initializes and runs the model orchestrator.
    """
    print("=" * 60)
    print("CVAT Custom Model Orchestrator")
    print("=" * 60)

    # Ensure models directory exists
    models_dir = os.path.join(project_root, "src", "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Start orchestrator
    orchestrator = ModelOrchestrator(models_dir=models_dir)
    orchestrator.run()


if __name__ == "__main__":
    main()
