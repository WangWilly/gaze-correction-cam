# Gaze Correction Camera

## Overview

This project implements a gaze correction system for video communication that uses computer vision and deep learning techniques to adjust eye gaze direction in real-time, providing a more natural eye contact experience during video calls.

## Prerequisites

The following dependencies are required to run this application:

- [Python 3.12+](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/) for dependency management
- [CMake](https://cmake.org/download/) (required for building dlib)
- [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/) (required for certain dependencies)

## Installation

1. Install system dependencies:

   ```bash
   brew install pkg-config
   brew install cmake
   ```

2. Install Python dependencies using Poetry:

   ```bash
   poetry install
   ```

3. Ensure model weights are downloaded and placed in the `weights/` directory.

4. Verify the face landmarks predictor file exists at `lm_feat/shape_predictor_68_face_landmarks.dat`.

## Configuration

The system parameters can be customized in the `config.py` file, including camera position, screen dimensions, and focal length.

## Usage

To start the gaze correction system:

```bash
python regz_socket_MP_FD.py
```

### Controls

- Press `q` to exit the application

## System Requirements

- macOS with camera access permissions
- Sufficient GPU resources for real-time processing

## References

The implementation is based on research in gaze correction techniques using warping-based convolutional neural networks.
