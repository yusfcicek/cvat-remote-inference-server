# -*- coding: utf-8 -*-
"""
Model runner service.

This module provides the entry point for running individual
model servers. It is invoked by the orchestrator as a subprocess.

Usage:
    python -m src.services.model_runner --model-name yolov12n --port 5001
"""

import argparse
import os
import sys
import signal
import gc

try:
    import torch
except ImportError:
    torch = None

import uvicorn

from src.api.app import (
    create_app,
    set_model_instance,
    set_idle_timeout,
    update_access_time
)
from src.services.model_loader import LazyModelWrapper


def main() -> None:
    """
    Main entry point for model runner.

    Parses command line arguments and starts the FastAPI server
    for a specific model.
    """
    parser = argparse.ArgumentParser(
        description="Run a model server for inference"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Instance name of the model"
    )
    parser.add_argument(
        "--implementation",
        help="Name of the model folder in src/models/ (defaults to model-name)"
    )
    parser.add_argument(
        "--model-config",
        help="JSON-encoded generic model configuration"
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Idle timeout in seconds before model unload"
    )

    args = parser.parse_args()

    # Determine implementation name
    impl_name = args.implementation or args.model_name

    # Parse model config
    import json
    model_config = {}
    if args.model_config:
        try:
            model_config = json.loads(args.model_config)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse model-config JSON: {e}")
            sys.exit(1)

    # Validate model directory exists
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    target_dir = os.path.join(project_root, "src", "models", impl_name)

    if not os.path.exists(target_dir):
        print(f"Error: Model directory {target_dir} does not exist.")
        sys.exit(1)

    # Configure model
    set_idle_timeout(args.timeout)
    model_instance = LazyModelWrapper(target_dir, args.model_name, model_config=model_config)
    set_model_instance(model_instance)
    update_access_time()

    print(
        f"Starting server for {args.model_name} "
        f"on port {args.port} with timeout {args.timeout}s"
    )

    # Create and run app
    app = create_app(
        title=f"{args.model_name} Inference API",
        description=f"Inference API for {args.model_name} model"
    )

    def shutdown_handler(signum, frame):
        print(f"\n[{args.model_name}] Received signal {signum}. Cleaning up resources...")
        
        # 1. Garbage Collection
        gc.collect()
        
        # 2. CUDA Cache (if torch is available)
        if torch and hasattr(torch, "cuda") and torch.cuda.is_available():
            print(f"[{args.model_name}] Clearing CUDA cache...")
            torch.cuda.empty_cache()
            
        print(f"[{args.model_name}] Cleanup complete. Exiting.")
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
