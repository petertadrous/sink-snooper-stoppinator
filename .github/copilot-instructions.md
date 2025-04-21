Sink Snooper Stoppinator

This project is an automated system designed to detect and deter cats from entering kitchen sinks. It uses computer vision and various deterrent mechanisms to create a pet-friendly but effective solution for keeping cats away from unwanted areas.

## Technologies Used
- **Python**: Main programming language
- **uv**: Virtual environment manager and package manager
- **pytest**: Unit testing framework
- **YOLO (You Only Look Once)**: Object detection system using YOLOv8
- **OpenCV**: For camera input and image processing
- **ONNX Runtime**: For running the YOLO model
- **GPIO**: For hardware-based deterrent mechanisms
- **PyAudio**: For audio-based deterrent system

## Best Practices

### Python Best Practices
1. **Project Structure**
   - Use modular architecture with clear separation of concerns
   - Follow Python package structure conventions
   - Keep configuration separate from code

2. **Code Quality**
   - Follow PEP 8 style guide
   - Use type hints for better code maintainability
   - Implement comprehensive unit tests
   - Use virtual environments for dependency management

3. **Testing**
   - Write unit tests for each module
   - Use pytest for testing framework
   - Test in virtual env using `uv run pytest --cov=src --cov-report=term-missing tests -v`
   - Maintain high test coverage, above 80% for each file
   - Mock hardware dependencies in tests
   - Mock sparingly and intentionally
   - Re-run and update unit tests after every new feature/enhancement

4. **Computer Vision**
   - Preprocess images for consistent input
   - Implement proper error handling for camera failures
   - Use appropriate thresholds for detection confidence

5. **Hardware Integration**
   - Implement proper cleanup for GPIO resources
   - Handle hardware failures gracefully
   - Use proper error handling for audio devices


### UV Best Practices
`uv` is a virtual environment manager. Commands are:
   - `uv add ...` to add packages to the virtual environment
   - `uv remove ...` to remove packages
   - `uv run ...` to run a command in the virtual environment
   - `uv run pytest --cov=src --cov-report=term-missing tests -v` to run unit tests

## Project Structure
```
├── assets/                     # Resource files
│   ├── audio files
│   ├── model weights
│   └── configuration files
├── src/                       # Source code
│   ├── detection/            # Computer vision and detection
│   ├── deterrent/            # Deterrent mechanisms
│   ├── models/              # ML model configurations
│   └── utils/              # Utility functions
├── tests/                  # Test suite
│   └── src/               # Tests mirroring src structure
├── main.py                # Application entry point
├── pyproject.toml         # Project configuration
└── uv.lock               # Dependency lock file
```
