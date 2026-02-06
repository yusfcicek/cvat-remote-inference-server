# Testing Framework

This directory contains automated tests for the CVAT Custom Model Orchestrator.

## Structure

- `test_api.py`: Tests for FastAPI endpoints, request/response cycle, and image processing utilities.
- `test_orchestrator.py`: Tests for the model orchestration logic, configuration management, and model lifecycle.

## Running Tests

To run the full test suite, use `pytest` from the project root:

```bash
source venv/bin/activate
pytest tests/
```

## What to Expect & Future Development

### 1. Robustness
Current tests focus on unit testing core components with mocks. As the project grows, we expect:
- **Integration Tests**: Running the orchestrator against a real filesystem with dummy implementation files.
- **Inference Validation**: Tests that use real weights (if available) to verify end-to-end detection results.

### 2. Multi-Instance Logic
With the new multi-instance support, future tests should ensure:
- Port conflict resolution works as expected.
- Multiple instances of the same model do not interfere with each other's memory or process state.

### 3. Configuration Flexibility
The transition to a generic `config: {}` block allows for many parameters. Tests should be added for:
- Validation of required keys within the generic config (using dynamic schemas or model-specific validators).
- Correct propagation of custom keys from `models.yaml` to the model `__init__` method.

## Code Quality
All tests should follow PEP8 standards and include descriptive docstrings explaining the scenario being tested.
