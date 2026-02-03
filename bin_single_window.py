#!/usr/bin/env python3
"""
Single Window Gaze Correction Application

A simplified gaze correction implementation using a single window.

Usage:
    python bin_single_window.py                      # Use dlib backend
    python bin_single_window.py --backend mediapipe  # Use mediapipe backend
    python bin_single_window.py --camera 1           # Use camera device 1

Controls:
    - 'g': Toggle gaze correction on/off
    - 'z': Auto-calibrate (look at camera, press z) - works anytime
    - 'c': Toggle calibration mode (WASD for manual adjustments)
    - 'f': Toggle FPS display
    - 'v': Toggle vector visualization
    - 'l': Toggle iris landmark visualization
    - 'm': Toggle face mesh visualization
    - 'q': Quit
"""

import sys
import cv2


def configure_tensorflow_device(device: str) -> str:
    """
    Configure TensorFlow to use specified device.
    Must be called before importing TensorFlow.

    Args:
        device: 'auto', 'cpu', or 'gpu'

    Returns:
        Actual device being used
    """
    import tensorflow as tf

    if device == "cpu":
        # Hide all GPUs to force CPU
        tf.config.set_visible_devices([], 'GPU')
        print("TensorFlow: Forcing CPU mode")
        return "CPU"
    elif device == "gpu":
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"TensorFlow: Using GPU ({gpus[0].name})")
            return "GPU"
        else:
            print("TensorFlow: No GPU found, falling back to CPU")
            return "CPU"
    else:  # auto
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"TensorFlow: Auto-detected GPU ({gpus[0].name})")
            return "GPU"
        else:
            print("TensorFlow: Auto-detected CPU only")
            return "CPU"


# Defer imports that depend on TensorFlow
def get_imports():
    from displayers.dis_single_window import SingleWindowGazeCorrector, DisplayConfig
    from displayers.face_predictor import create_face_predictor
    return SingleWindowGazeCorrector, DisplayConfig, create_face_predictor


def detect_camera_resolution(camera_id: int) -> tuple[int, int]:
    """
    Detect the actual resolution of the specified camera.
    
    Args:
        camera_id: Camera device ID
        
    Returns:
        Tuple of (width, height) in pixels
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Warning: Could not open camera {camera_id}, using default resolution")
        return (640, 480)
    
    # Get the actual resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Detected camera resolution: {width}x{height}")
    return (width, height)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Single Window Gaze Correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  'g' - Toggle gaze correction on/off
  'z' - Auto-calibrate (look at camera, press z) - works anytime
  'c' - Toggle calibration mode (WASD for manual adjustments)
  'f' - Toggle FPS display
  'v' - Toggle vector visualization
  'l' - Toggle iris landmark visualization
  'm' - Toggle face mesh visualization
  'q' - Quit the application

Examples:
  %(prog)s                         # Use default dlib backend
  %(prog)s --backend mediapipe     # Use MediaPipe for face detection
  %(prog)s --camera 1              # Use camera device 1
  %(prog)s --device cpu            # Force CPU mode (for performance comparison)
        """,
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="dlib",
        choices=["dlib", "mediapipe"],
        help="Face detection backend (default: dlib)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./model_managers/gaze_corrector_v1_01.yaml",
        help="Path to gaze corrector config file (default: ./model_managers/gaze_corrector_v1_01.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="TensorFlow device: auto, cpu, or gpu (default: auto)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug output for v2 gaze estimation",
    )
    args = parser.parse_args()

    # Configure TensorFlow device BEFORE importing TF-dependent modules
    device_used = configure_tensorflow_device(args.device)

    # Now import modules that depend on TensorFlow
    SingleWindowGazeCorrector, DisplayConfig, create_face_predictor = get_imports()

    # Detect camera resolution
    video_size = detect_camera_resolution(args.camera)

    # Calculate appropriate face detection size (half resolution)
    face_detect_size = (video_size[0] // 2, video_size[1] // 2)

    # Create display config with detected resolution
    display_config = DisplayConfig(
        video_size=video_size,
        face_detect_size=face_detect_size,
    )

    print(f"Video size: {video_size}, Face detection size: {face_detect_size}")
    print(f"TensorFlow device: {device_used}")

    # Create face predictor based on selected backend
    predictor = create_face_predictor(args.backend)

    # Create and run the corrector
    corrector = SingleWindowGazeCorrector(
        face_predictor=predictor,
        display_config=display_config,
        camera_id=args.camera,
        config_path=args.config,
        device_name=device_used,
        verbose=args.verbose,
    )
    corrector.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        sys.exit(0)
