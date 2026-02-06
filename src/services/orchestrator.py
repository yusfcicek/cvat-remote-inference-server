# -*- coding: utf-8 -*-
"""
Model orchestrator service with auto-detection.

This module provides the ModelOrchestrator class which manages
multiple model servers, starting and stopping them based on
configuration and file system state.

Features:
    - Auto-detects new model directories (no restart required)
    - Prioritizes YAML configuration for ports
    - Auto-registers new models with next available port if undefined
    - Auto-generates Nuclio functions for new models
    - Watches for file changes in real-time
    - Restarts crashed servers automatically
"""

import os
import shutil
import subprocess
import sys
import signal
import time
from typing import Any, Dict, Optional, Set

from src.core.config import (
    Settings,
    get_settings,
    reload_settings,
    get_config_path,
    add_model_to_config,
    get_next_available_port,
)
from src.utils.model_utils import detect_model_type_from_dir


class ModelOrchestrator:
    """
    Orchestrator for managing multiple model server processes.

    This class monitors the configuration and model directories,
    automatically starting and stopping model servers as needed.
    New models are detected automatically without requiring restart.

    Attributes:
        models_dir: Directory containing model implementations.
        process_registry: Dictionary mapping model names to Popen instances.
        poll_interval: Seconds between sync operations.
        nuclio_output_dir: Directory for generated Nuclio functions.
        known_models: Set of previously detected models.
    """

    def __init__(
        self,
        models_dir: str = "src/models",
        poll_interval: int = 5,
        nuclio_output_dir: Optional[str] = None
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            models_dir: Path to models directory relative to project root.
            poll_interval: Seconds between configuration sync operations.
            nuclio_output_dir: Directory for generated Nuclio functions.
        """
        self.models_dir = models_dir
        self.poll_interval = poll_interval
        
        if nuclio_output_dir:
            self.nuclio_output_dir = nuclio_output_dir
        else:
            self.nuclio_output_dir = get_settings().cvat.nuclio_output_dir
        self.process_registry: Dict[str, subprocess.Popen] = {}
        self.known_models: Set[str] = set()

    def _scan_model_directories(self) -> Set[str]:
        """
        Scan for valid model directories in the models folder.

        A valid model directory contains one of the supported implementation files.

        Returns:
            Set of valid model directory names.
        """
        valid_models = set()

        if not os.path.exists(self.models_dir):
            return valid_models

        for entry in os.listdir(self.models_dir):
            model_path = os.path.join(self.models_dir, entry)

            # Skip non-directories and special directories
            if not os.path.isdir(model_path):
                continue
            if entry.startswith("_") or entry.startswith("."):
                continue

            # Use shared utility for validation
            if detect_model_type_from_dir(model_path):
                valid_models.add(entry)

        return valid_models

    def _register_if_needed(self, model_name: str, settings: Settings) -> bool:
        """
        Register model if it's missing from config or has no port.

        If the model exists in config but has no port, it assigns one.
        If the model is completely new, it assigns a port and adds it.

        Args:
            model_name: Name of the model.
            settings: Current settings.

        Returns:
            True if changes were made to config, False otherwise.
        """
        # Case 1: Model exists in config
        if model_name in settings.models:
            model_config = settings.models[model_name]
            if model_config.port is not None:
                return False  # Already registered and has a port

            print(f"[Orchestrator] Model {model_name} in config but missing port.")
        else:
            # Case 2: Model needs registration
            print(f"[Orchestrator] Model {model_name} detected but not configured.")

        try:
            port = get_next_available_port()
            # Preserve existing settings if any
            idle_timeout = 300
            if model_name in settings.models:
                m_cfg = settings.models[model_name]
                idle_timeout = m_cfg.idle_timeout_seconds

            add_model_to_config(
                model_name,
                port,
                idle_timeout_seconds=idle_timeout,
                implementation=model_name
            )
            print(f"[Orchestrator] Assigned port {port} to {model_name}")
            return True
        except Exception as e:
            print(f"[Orchestrator] Failed to register {model_name}: {e}")
            return False

    def _is_function_deployed(self, name: str) -> bool:
        """Check if a function is already deployed in the cluster."""
        try:
            # Explicitly target local platform as used in CVAT serverless
            cmd = ["nuctl", "get", "function", name, "--platform", "local"]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=False
            )
            # If name is in output and return code is 0, it exists and is ready
            return result.returncode == 0 and name in result.stdout
        except Exception as e:
            print(f"[Orchestrator] Error checking deployment status for {name}: {e}")
            return False

    def _generate_nuclio_function(self, model_name: str, force_deploy: bool = False) -> None:
        """Generate Nuclio function files for a model and deploy them."""
        try:
            # 1. Determine paths
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            deploy_script = os.path.join(project_root, "serverless", "deploy.sh")
            
            # Resolve the output directory to an absolute path
            output_dir = os.path.abspath(self.nuclio_output_dir)
            model_nuclio_dir = os.path.join(output_dir, model_name)

            # 2. Check if already deployed to avoid redundant work
            if not force_deploy and self._is_function_deployed(model_name):
                print(f"[Orchestrator] Model {model_name} is already deployed to CVAT. Skipping deployment.")
                return

            # Import here to avoid circular imports
            from scripts.generate_nuclio_function import generate_function

            # Create output directory surgically
            os.makedirs(output_dir, exist_ok=True)
            if os.path.exists(model_nuclio_dir):
                shutil.rmtree(model_nuclio_dir)

            # Generate files
            print(f"[Orchestrator] Generating Nuclio files for {model_name} in {model_nuclio_dir}...")
            generate_function(model_name, output_dir)
            print(f"[Orchestrator] Nuclio files generated successfully.")

            # 3. Deploy to CVAT (Container part)
            if os.path.exists(deploy_script):
                print(f"[Orchestrator] Deploying {model_name} container to CVAT via {deploy_script}...")
                try:
                    # Run the deployment script targeting the specific model folder
                    deploy_cmd = ["bash", deploy_script, model_nuclio_dir]
                    result = subprocess.run(
                        deploy_cmd, 
                        capture_output=True, 
                        text=True, 
                        check=False
                    )
                    
                    if result.returncode == 0:
                        print(f"[Orchestrator] CVAT deployment successful for {model_name}")
                    else:
                        print(f"[Orchestrator] CVAT deployment failed for {model_name} (exit code: {result.returncode})")
                    
                    if result.stdout:
                        print(f"[Orchestrator] Deployment stdout:\n{result.stdout.strip()}")
                    if result.stderr:
                        print(f"[Orchestrator] Deployment stderr:\n{result.stderr.strip()}")
                        
                except Exception as e:
                    print(f"[Orchestrator] Failed to execute deployment script for {model_name}: {e}")
            else:
                print(f"[Orchestrator] Deployment script NOT FOUND at {deploy_script}. Skipping container deployment.")

        except Exception as e:
            print(f"[Orchestrator] Failed to manage Nuclio for {model_name}: {e}")
            import traceback
            traceback.print_exc()

    def start_model(
        self,
        name: str,
        port: int,
        timeout: int,
        implementation: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Start a model server subprocess.

        Args:
            name: Instance name of the model.
            port: Port to bind the server to.
            timeout: Idle timeout in seconds.
            implementation: Name of the model directory.
            model_config: Generic configuration for the model.
        """
        impl_name = implementation or name

        # Get python interpreter
        settings = get_settings()
        python_exe = sys.executable
        if name in settings.models:
            custom_python = settings.models[name].interpreter_path
            if custom_python:
                python_exe = custom_python

        print(f"[Orchestrator] Starting {name} (impl: {impl_name}) on port {port}...")
        if python_exe != sys.executable:
            print(f"[Orchestrator] Using custom interpreter: {python_exe}")

        cmd = [
            python_exe, "-m", "src.services.model_runner",
            "--model-name", name,
            "--port", str(port),
            "--timeout", str(timeout)
        ]

        if implementation:
            cmd.extend(["--implementation", implementation])
        
        if model_config:
            import json
            cmd.extend(["--model-config", json.dumps(model_config)])

        proc = subprocess.Popen(cmd)
        self.process_registry[name] = proc

    def stop_model(self, name: str, permanent: bool = False) -> None:
        """
        Stop a model server subprocess.
        
        Args:
            name: Instance name of the model.
            permanent: If True, also delete from CVAT and cleanup files.
        """
        proc = self.process_registry.get(name)

        if proc:
            print(f"[Orchestrator] Stopping {name}...")
            proc.terminate()

            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

            del self.process_registry[name]

        if not permanent:
            return

        # Permanent removal cleanup
        # 1. Delete from Nuclio
        print(f"[Orchestrator] Deleting Nuclio function {name} from CVAT (Permanent Removal)...")
        try:
            delete_cmd = ["nuctl", "delete", "function", name, "--platform", "local"]
            result = subprocess.run(
                delete_cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                print(f"[Orchestrator] Nuclio function {name} deleted successfully")
            else:
                print(f"[Orchestrator] Failed to delete Nuclio function {name} (exit code: {result.returncode})")
                
            if result.stdout:
                print(f"[Orchestrator] Nuctl delete stdout:\n{result.stdout.strip()}")
            if result.stderr:
                print(f"[Orchestrator] Nuctl delete stderr:\n{result.stderr.strip()}")
        except Exception as e:
            print(f"[Orchestrator] Failed to execute nuctl delete for {name}: {e}")

        # 2. Delete the specific model subfolder in nuclio_output_dir (Surgical)
        model_nuclio_dir = os.path.join(self.nuclio_output_dir, name)
        if os.path.exists(model_nuclio_dir):
            print(f"[Orchestrator] Cleaning up Nuclio deployment folder: {model_nuclio_dir}")
            try:
                shutil.rmtree(model_nuclio_dir)
            except Exception as e:
                print(f"[Orchestrator] Failed to delete folder {model_nuclio_dir}: {e}")

    def sync_processes(self) -> None:
        """
        Synchronize running processes with configuration.

        Flow:
        1. Scan file system for available models
        2. Check against current config (YAML)
        3. Assign ports to unconfigured models and update YAML
        4. Generate Nuclio functions for new models
        5. Start/Restart/Stop servers to match state
        """
        # 1. Scan for physical model directories
        detected_models = self._scan_model_directories()

        # 2. Reload settings to get current YAML state
        settings = reload_settings()
        self.nuclio_output_dir = settings.cvat.nuclio_output_dir
        config_changed = False

        # 3. Register models that need ports (detected or pre-configured)
        all_model_names = detected_models.union(settings.models.keys())
        for model_name in all_model_names:
            self._register_if_needed(model_name, settings)

        # Reload settings to get updated state after registration
        settings = reload_settings()
        self.nuclio_output_dir = settings.cvat.nuclio_output_dir

        # 4. Generate Nuclio functions for all configured instances
        for instance_name, config in settings.models.items():
            # Check if Nuclio function generation is needed
            nuclio_instance_dir = os.path.join(self.nuclio_output_dir, instance_name)
            main_py_path = os.path.join(nuclio_instance_dir, "main.py")
            
            needs_generation = not os.path.exists(nuclio_instance_dir) or not os.path.exists(main_py_path)
            
            if not needs_generation:
                # Check if config is newer than generated function
                config_path = get_config_path()
                if os.path.exists(config_path):
                    if os.path.getmtime(config_path) > os.path.getmtime(main_py_path):
                        needs_generation = True

            if not needs_generation:
                # Even if files are fine, check if it's actually deployed in CVAT
                if not self._is_function_deployed(instance_name):
                    print(f"[Orchestrator] {instance_name} files exist but function is not in CVAT. Redeploying...")
                    needs_generation = True

            if needs_generation:
                print(f"[Orchestrator] Generating Nuclio function for {instance_name}...")
                self._generate_nuclio_function(instance_name, force_deploy=True)

        # 5. Determine which models should be running
        # Only start models that exist both physically AND in config
        valid_models = set()
        for name, config in settings.models.items():
            impl_name = config.implementation or name
            impl_path = os.path.join(self.models_dir, impl_name)
            
            # Use shared utility to check if implementation exists physically
            if detect_model_type_from_dir(impl_path):
                valid_models.add(name)
            else:
                print(f"[Orchestrator] Model {name} implementation {impl_name} not found at {impl_path}")

        # 5. Start/Restart valid models
        for name in valid_models:
            model_config = settings.models[name]

            if name not in self.process_registry:
                # Start new
                self.start_model(
                    name,
                    model_config.port,
                    model_config.idle_timeout_seconds,
                    implementation=model_config.implementation,
                    model_config=model_config.config
                )
            else:
                # Check health
                if self.process_registry[name].poll() is not None:
                    print(f"[Orchestrator] {name} process died. Restarting...")
                    self.start_model(
                        name,
                        model_config.port,
                        model_config.idle_timeout_seconds,
                        implementation=model_config.implementation,
                        model_config=model_config.config
                    )

        # 6. Stop removed models
        running_names = list(self.process_registry.keys())
        for name in running_names:
            if name not in valid_models:
                # This is a removal, so clean up CVAT and local files
                self.stop_model(name, permanent=True)

    def run(self) -> None:
        """Run the orchestrator main loop."""
        print("=" * 60)
        print("[Orchestrator] Starting with auto-detection enabled")
        print(f"[Orchestrator] Watching: {self.models_dir}/")
        print(f"[Orchestrator] Nuclio output: {self.nuclio_output_dir}/")
        print("=" * 60)

        # Flag for graceful shutdown
        self._running = True

        def signal_handler(signum, frame):
            sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            print(f"\n[Orchestrator] Received {sig_name}. Shutting down...")
            self._running = False

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while self._running:
                self.sync_processes()
                
                # Sleep in small increments to allow for faster signal response
                for _ in range(self.poll_interval):
                    if not self._running:
                        break
                    time.sleep(1)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Stop all running model servers."""
        for name in list(self.process_registry.keys()):
            self.stop_model(name)


def main() -> None:
    """Entry point."""
    orchestrator = ModelOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
