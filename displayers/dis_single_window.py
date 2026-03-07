"""
Single Window Gaze Corrector

A simplified gaze correction application that:
- Uses a single window (no socket communication)
- Allows real-time toggle of gaze correction with 'g' key
- Supports injectable face predictor backends
- Decoupled architecture: FacePredictor -> GazeCorrector
- Calibration mode for camera offset adjustment
"""

import time
import cv2
from dataclasses import dataclass
from typing import Optional

from utils.logger import Logger
from displayers.face_predictor import (
    FacePredictor,
    EyeExtractionConfig,
    create_face_predictor,
)
from model_managers.gaze_corrector_v1 import GazeCorrector


################################################################################
# Configuration
################################################################################


@dataclass
class DisplayConfig:
    """Configuration for the display application."""

    video_size: tuple[int, int] = (640, 480)
    face_detect_size: tuple[int, int] = (320, 240)
    window_name: str = "Gaze Correction"

    @property
    def x_ratio(self) -> float:
        return self.video_size[0] / self.face_detect_size[0]

    @property
    def y_ratio(self) -> float:
        return self.video_size[1] / self.face_detect_size[1]


@dataclass
class CalibrationConfig:
    """Configuration for calibration mode."""

    step_xy: float = 0.5      # cm per key press for X/Y
    step_z: float = 0.5       # cm per key press for Z
    step_focal: float = 10.0  # pixels per key press for focal length
    step_gaze: float = 1.0    # degrees per key press for gaze offset (WASD)
    step_scale: float = 2.0   # degrees per key press for gaze scale (E/Q)


################################################################################
# Single Window Gaze Corrector
################################################################################


class SingleWindowGazeCorrector:
    """
    Single-window gaze correction application with real-time toggle.

    This class orchestrates:
    - FacePredictor: Detects faces and extracts eye data
    - GazeCorrector: Applies gaze correction model

    Controls:
        - 'g': Toggle gaze correction on/off
        - 'z': Auto-calibrate (look at camera, press z to zero) - works anytime
        - 'c': Toggle calibration mode (for manual adjustments)
        - 'f': Toggle FPS display
        - 'v': Toggle vector visualization
        - 'l': Toggle iris landmark visualization
        - 'm': Toggle face mesh visualization
        - 'q': Quit

    Calibration Mode Controls:
        - WASD: Adjust manual gaze offset
            - W/S: Vertical offset (up/down)
            - A/D: Horizontal offset (left/right)
        - E/Q: Adjust gaze scale (sensitivity)
        - Arrow keys: Adjust camera X/Y offset (legacy geometric mode)
        - '+'/'-': Adjust Z offset
        - '['/']': Adjust focal length
        - 'r': Reset all to defaults
    """

    # Arrow key codes (platform-dependent)
    # KEY_UP = 82
    # KEY_DOWN = 84
    # KEY_LEFT = 81
    # KEY_RIGHT = 83
    KEY_UP = 0
    KEY_DOWN = 1
    KEY_LEFT = 2
    KEY_RIGHT = 3

    def __init__(
        self,
        face_predictor: Optional[FacePredictor] = None,
        gaze_corrector: Optional[GazeCorrector] = None,
        display_config: Optional[DisplayConfig] = None,
        calibration_config: Optional[CalibrationConfig] = None,
        camera_id: int = 0,
        config_path: str = "./model_managers/gaze_corrector_v1_01.yaml",
        device_name: str = "auto",
        verbose: bool = False,
    ):
        self.logger = Logger(self.__class__.__name__)
        self.display_cfg = display_config or DisplayConfig()
        self.calib_cfg = calibration_config or CalibrationConfig()

        # Initialize face predictor (injectable)
        self.face_predictor = face_predictor or create_face_predictor("dlib")
        self.logger.log(f"Using face predictor: {self.face_predictor.get_name()}")

        # Initialize gaze corrector (injectable)
        self.gaze_corrector = gaze_corrector or GazeCorrector(
            config_path=config_path,
            verbose=verbose,
        )

        # Eye extraction config (matches model requirements)
        self.eye_config = EyeExtractionConfig()

        # State
        self.gaze_correction_enabled = True
        self.calibration_mode = False
        self.vector_visualization_enabled = False
        self.landmark_visualization_enabled = False
        self.face_mesh_visualization_enabled = False
        self.fps_display_enabled = False
        self.camera_id = camera_id
        self.device_name = device_name

        # Last face data for landmark visualization
        self.last_face_data = None

        # FPS tracking
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()

        # Store default values for reset
        self.default_camera_offset = self.gaze_corrector.get_camera_offset()
        self.default_focal_length = self.gaze_corrector.get_focal_length()

    def draw_status(self, frame) -> None:
        """Draw status overlay on frame."""
        status = "GAZE ON" if self.gaze_correction_enabled else "GAZE OFF"
        color = (0, 255, 0) if self.gaze_correction_enabled else (0, 0, 255)

        box_height = 60
        if self.vector_visualization_enabled:
            box_height += 14
        if self.landmark_visualization_enabled:
            box_height += 14
        if self.face_mesh_visualization_enabled:
            box_height += 14
        if self.fps_display_enabled:
            box_height += 14
        cv2.rectangle(frame, (10, 10), (150, box_height), (0, 0, 0), -1)
        cv2.putText(
            frame, status, (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )
        # Show backend and device
        backend_device = f"[{self.face_predictor.get_name()}|{self.device_name}]"
        cv2.putText(
            frame, backend_device, (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA
        )

        # Vector and landmark visualization status
        y_pos = 70
        if self.vector_visualization_enabled:
            cv2.putText(
                frame, "VECTORS", (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA
            )
            y_pos += 14
        if self.landmark_visualization_enabled:
            cv2.putText(
                frame, "IRIS", (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA
            )
            y_pos += 14
        if self.face_mesh_visualization_enabled:
            cv2.putText(
                frame, "FACE MESH", (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA
            )
            y_pos += 14

        # FPS display
        if self.fps_display_enabled:
            cv2.putText(
                frame, f"FPS: {self.fps:.1f}", (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA
            )

    def draw_calibration_overlay(self, frame) -> None:
        """Draw calibration mode overlay with gaze offset and camera offset visualization."""
        h, w = frame.shape[:2]

        # Semi-transparent overlay background (size adjusts for head pose)
        overlay = frame.copy()
        box_height = 375 if self.gaze_corrector.last_head_pose is not None else 345
        cv2.rectangle(overlay, (w - 260, 10), (w - 10, box_height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title with calibration status
        is_calibrated = self.gaze_corrector.is_calibrated()
        status_color = (0, 255, 0) if is_calibrated else (100, 100, 255)
        status_text = "CALIBRATED" if is_calibrated else "NOT CALIBRATED"
        cv2.putText(
            frame, "CALIBRATION MODE", (w - 250, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"[{status_text}]", (w - 250, 52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1, cv2.LINE_AA
        )

        # Auto-calibrate hint
        cv2.putText(
            frame, "Press Z to calibrate (look at cam)", (w - 250, 72),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1, cv2.LINE_AA
        )

        # Manual gaze offset (WASD)
        gaze_v, gaze_h = self.gaze_corrector.get_gaze_offset()
        cv2.putText(
            frame, "Gaze Offset (WASD):", (w - 250, 97),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"  V: {gaze_v:+.1f} deg   H: {gaze_h:+.1f} deg", (w - 250, 115),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 200), 1, cv2.LINE_AA
        )

        # Gaze scale (E/Q)
        gaze_scale = self.gaze_corrector.get_gaze_scale()
        cv2.putText(
            frame, f"Gaze Scale (E/Q): {gaze_scale:.1f} deg", (w - 250, 137),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1, cv2.LINE_AA
        )

        # Current camera offset (secondary - arrows)
        offset = self.gaze_corrector.get_camera_offset()
        cv2.putText(
            frame, f"Camera Offset (Arrows):", (w - 250, 162),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"  X: {offset[0]:+.1f}  Y: {offset[1]:+.1f}  Z: {offset[2]:+.1f} cm", (w - 250, 179),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 200), 1, cv2.LINE_AA
        )

        # Eye position (estimated)
        eye_pos = self.gaze_corrector.get_last_eye_position()
        cv2.putText(
            frame, f"Eye Position:", (w - 250, 202),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"  X: {eye_pos[0]:+.1f}  Y: {eye_pos[1]:+.1f}  Z: {eye_pos[2]:+.1f} cm", (w - 250, 219),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 200), 1, cv2.LINE_AA
        )

        # Focal length
        focal = self.gaze_corrector.get_focal_length()
        cv2.putText(
            frame, f"Focal Length: {focal:.0f} px", (w - 250, 242),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 180, 200), 1, cv2.LINE_AA
        )

        # Detected gaze (live)
        detected_v, detected_h = self.gaze_corrector.last_detected_gaze
        cv2.putText(
            frame, f"Detected Gaze:", (w - 250, 267),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"  V: {detected_v:+.1f} deg   H: {detected_h:+.1f} deg", (w - 250, 284),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 150, 150), 1, cv2.LINE_AA
        )

        # Head pose (if available)
        head_pose = self.gaze_corrector.last_head_pose
        if head_pose is not None:
            pitch, yaw, roll = head_pose
            cv2.putText(
                frame, f"Head Pose:", (w - 250, 304),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA
            )
            cv2.putText(
                frame, f"  P:{pitch:+.0f} Y:{yaw:+.0f} R:{roll:+.0f}", (w - 250, 321),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 200, 255), 1, cv2.LINE_AA
            )
            controls_y = 346
        else:
            controls_y = 312

        # Controls hints
        cv2.putText(
            frame, "[Z:Auto] [WASD:Offset] [E/Q:Scale]", (w - 250, controls_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, "[Arrows:Cam] [+/-:Depth] [R:Reset]", (w - 250, controls_y + 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1, cv2.LINE_AA
        )

        # Draw camera position diagram (bottom left)
        self._draw_camera_diagram(frame, offset, eye_pos)

    def _draw_camera_diagram(
        self, frame, camera_offset: tuple, eye_pos: list
    ) -> None:
        """Draw a simple diagram showing camera and eye positions."""
        h, w = frame.shape[:2]

        # Diagram area
        diagram_x, diagram_y = 10, h - 160
        diagram_w, diagram_h = 150, 150

        # Background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (diagram_x, diagram_y),
            (diagram_x + diagram_w, diagram_y + diagram_h),
            (30, 30, 30), -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(
            frame, "Top View (X-Z)", (diagram_x + 10, diagram_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA
        )

        # Center point (screen center, Z=0)
        center_x = diagram_x + diagram_w // 2
        center_y = diagram_y + diagram_h - 30

        # Scale: 1cm = 2 pixels
        scale = 2

        # Draw screen line
        cv2.line(
            frame,
            (diagram_x + 10, center_y),
            (diagram_x + diagram_w - 10, center_y),
            (100, 100, 100), 2
        )
        cv2.putText(
            frame, "Screen", (diagram_x + 50, center_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1, cv2.LINE_AA
        )

        # Draw camera position (offset from screen center)
        cam_px = center_x + int(camera_offset[0] * scale)
        cam_py = center_y + int(camera_offset[2] * scale)  # Z goes up in diagram
        cv2.circle(frame, (cam_px, cam_py), 6, (0, 255, 255), -1)
        cv2.putText(
            frame, "Cam", (cam_px - 12, cam_py - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA
        )

        # Draw eye position (estimated)
        eye_px = center_x + int(eye_pos[0] * scale)
        eye_py = center_y + int(eye_pos[2] * scale)
        cv2.circle(frame, (eye_px, eye_py), 5, (255, 100, 100), -1)
        cv2.putText(
            frame, "Eye", (eye_px - 10, eye_py - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 100), 1, cv2.LINE_AA
        )

        # Draw gaze line from eye to camera
        cv2.line(frame, (eye_px, eye_py), (cam_px, cam_py), (100, 255, 100), 1)

    def handle_calibration_key(self, key: int) -> bool:
        """
        Handle calibration mode key presses.

        Args:
            key: Key code

        Returns:
            True if key was handled
        """
        step_xy = self.calib_cfg.step_xy
        step_z = self.calib_cfg.step_z
        step_gaze = self.calib_cfg.step_gaze
        step_scale = self.calib_cfg.step_scale

        # E/Q: Adjust gaze scale (sensitivity)
        if key == ord("e"):
            new_scale = self.gaze_corrector.adjust_gaze_scale(step_scale)
            self.logger.log(f"Gaze scale: {new_scale:.1f}")
            return True
        elif key == ord("q"):
            new_scale = self.gaze_corrector.adjust_gaze_scale(-step_scale)
            self.logger.log(f"Gaze scale: {new_scale:.1f}")
            return True

        # WASD: Manual gaze offset (fine-tuning)
        elif key == ord("w"):
            # W = correct upward (if eyes appear to look down)
            self.gaze_corrector.adjust_gaze_offset(dv=step_gaze)
            return True
        elif key == ord("s"):
            # S = correct downward (if eyes appear to look up)
            self.gaze_corrector.adjust_gaze_offset(dv=-step_gaze)
            return True
        elif key == ord("a"):
            # A = correct leftward (if eyes appear to look right)
            self.gaze_corrector.adjust_gaze_offset(dh=-step_gaze)
            return True
        elif key == ord("d"):
            # D = correct rightward (if eyes appear to look left)
            self.gaze_corrector.adjust_gaze_offset(dh=step_gaze)
            return True

        # Arrow keys: Camera offset (legacy geometric mode)
        elif key == self.KEY_LEFT:
            self.gaze_corrector.adjust_camera_offset(dx=-step_xy)
            return True
        elif key == self.KEY_RIGHT:
            self.gaze_corrector.adjust_camera_offset(dx=step_xy)
            return True
        elif key == self.KEY_UP:
            self.gaze_corrector.adjust_camera_offset(dy=-step_xy)
            return True
        elif key == self.KEY_DOWN:
            self.gaze_corrector.adjust_camera_offset(dy=step_xy)
            return True
        elif key == ord("+") or key == ord("="):
            self.gaze_corrector.adjust_camera_offset(dz=step_z)
            return True
        elif key == ord("-") or key == ord("_"):
            self.gaze_corrector.adjust_camera_offset(dz=-step_z)
            return True
        elif key == ord("["):
            self.gaze_corrector.adjust_focal_length(-self.calib_cfg.step_focal)
            return True
        elif key == ord("]"):
            self.gaze_corrector.adjust_focal_length(self.calib_cfg.step_focal)
            return True
        elif key == ord("r"):
            # Reset all calibration values
            x, y, z = self.default_camera_offset
            self.gaze_corrector.set_camera_offset(x, y, z)
            self.gaze_corrector.set_focal_length(self.default_focal_length)
            self.gaze_corrector.reset_gaze_offset()
            self.gaze_corrector.reset_neutral_gaze()  # Reset v3 calibration
            self.gaze_corrector.set_gaze_scale(20.0)  # Reset to default
            self.logger.log("All calibration values reset to defaults")
            return True

        return False

    def process_frame(self, frame):
        """
        Process a single frame with gaze correction.

        Args:
            frame: BGR video frame

        Returns:
            Processed frame with gaze correction applied
        """
        display_frame = frame.copy()

        # Get eye data for all detected faces
        face_data_list = self.face_predictor.list_eye_data(frame, self.eye_config)

        # Process first detected face
        for face_data in face_data_list:
            try:
                # Store for landmark visualization
                self.last_face_data = face_data

                # Apply gaze correction (pass video_size)
                display_frame = self.gaze_corrector.apply_correction(
                    display_frame, face_data, self.display_cfg.video_size
                )

                # Draw vectors if enabled
                if self.vector_visualization_enabled:
                    display_frame = self.gaze_corrector.draw_vectors(display_frame)

                # Draw iris landmarks if enabled
                if self.landmark_visualization_enabled:
                    self.draw_iris_landmarks(display_frame, face_data)

                # Draw face mesh if enabled
                if self.face_mesh_visualization_enabled:
                    self.draw_face_mesh(display_frame, face_data)
            except Exception as e:
                import traceback
                self.logger.log(f"Error: {e}")
                traceback.print_exc()  # Print full traceback for debugging
            break  # Only process first face

        return display_frame

    def draw_iris_landmarks(self, frame, face_data) -> None:
        """
        Draw iris landmarks on the frame for debugging.

        Args:
            frame: BGR video frame to draw on
            face_data: FaceData containing eye information
        """
        import cv2

        # Colors for different points
        colors = [
            (0, 255, 0),    # Top - Green
            (255, 0, 0),    # Right - Blue
            (0, 0, 255),    # Bottom - Red
            (255, 255, 0),  # Left - Cyan
        ]
        labels = ["T", "R", "B", "L"]

        # Draw left eye iris landmarks
        if face_data.left_eye and face_data.left_eye.iris_points:
            for i, pt in enumerate(face_data.left_eye.iris_points):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(frame, (x, y), 4, colors[i], -1)
                cv2.putText(
                    frame, f"L{labels[i]}", (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[i], 1, cv2.LINE_AA
                )
            # Draw center
            cx = sum(p[0] for p in face_data.left_eye.iris_points) / 4
            cy = sum(p[1] for p in face_data.left_eye.iris_points) / 4
            cv2.circle(frame, (int(cx), int(cy)), 3, (255, 255, 255), -1)

        # Draw right eye iris landmarks
        if face_data.right_eye and face_data.right_eye.iris_points:
            for i, pt in enumerate(face_data.right_eye.iris_points):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(frame, (x, y), 4, colors[i], -1)
                cv2.putText(
                    frame, f"R{labels[i]}", (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[i], 1, cv2.LINE_AA
                )
            # Draw center
            cx = sum(p[0] for p in face_data.right_eye.iris_points) / 4
            cy = sum(p[1] for p in face_data.right_eye.iris_points) / 4
            cv2.circle(frame, (int(cx), int(cy)), 3, (255, 255, 255), -1)

    def draw_face_mesh(self, frame, face_data) -> None:
        """
        Draw face mesh landmarks on the frame for debugging.

        Args:
            frame: BGR video frame to draw on
            face_data: FaceData containing landmarks
        """
        if face_data.landmarks is None or face_data.landmarks.raw_shape is None:
            return

        h, w = frame.shape[:2]
        landmarks = face_data.landmarks.raw_shape

        # Check if this is a MediaPipe landmarks object (has indexable landmarks with x,y,z)
        try:
            # Draw key face contour points (subset for cleaner visualization)
            # Face oval: 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            #            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            #            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

            # Draw face oval
            for i in range(len(face_oval)):
                idx1, idx2 = face_oval[i], face_oval[(i + 1) % len(face_oval)]
                pt1 = (int(landmarks[idx1].x * w), int(landmarks[idx1].y * h))
                pt2 = (int(landmarks[idx2].x * w), int(landmarks[idx2].y * h))
                cv2.line(frame, pt1, pt2, (255, 0, 255), 1, cv2.LINE_AA)

            # Left eyebrow: 276, 283, 282, 295, 285
            left_eyebrow = [276, 283, 282, 295, 285]
            for i in range(len(left_eyebrow) - 1):
                pt1 = (int(landmarks[left_eyebrow[i]].x * w), int(landmarks[left_eyebrow[i]].y * h))
                pt2 = (int(landmarks[left_eyebrow[i+1]].x * w), int(landmarks[left_eyebrow[i+1]].y * h))
                cv2.line(frame, pt1, pt2, (0, 200, 200), 1, cv2.LINE_AA)

            # Right eyebrow: 46, 53, 52, 65, 55
            right_eyebrow = [46, 53, 52, 65, 55]
            for i in range(len(right_eyebrow) - 1):
                pt1 = (int(landmarks[right_eyebrow[i]].x * w), int(landmarks[right_eyebrow[i]].y * h))
                pt2 = (int(landmarks[right_eyebrow[i+1]].x * w), int(landmarks[right_eyebrow[i+1]].y * h))
                cv2.line(frame, pt1, pt2, (0, 200, 200), 1, cv2.LINE_AA)

            # Left eye contour: 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
            left_eye_contour = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            for i in range(len(left_eye_contour)):
                idx1, idx2 = left_eye_contour[i], left_eye_contour[(i + 1) % len(left_eye_contour)]
                pt1 = (int(landmarks[idx1].x * w), int(landmarks[idx1].y * h))
                pt2 = (int(landmarks[idx2].x * w), int(landmarks[idx2].y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)

            # Right eye contour: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
            right_eye_contour = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            for i in range(len(right_eye_contour)):
                idx1, idx2 = right_eye_contour[i], right_eye_contour[(i + 1) % len(right_eye_contour)]
                pt1 = (int(landmarks[idx1].x * w), int(landmarks[idx1].y * h))
                pt2 = (int(landmarks[idx2].x * w), int(landmarks[idx2].y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)

            # Nose: 168, 6, 197, 195, 5, 4, 1, 19, 94, 2
            nose = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
            for i in range(len(nose) - 1):
                pt1 = (int(landmarks[nose[i]].x * w), int(landmarks[nose[i]].y * h))
                pt2 = (int(landmarks[nose[i+1]].x * w), int(landmarks[nose[i+1]].y * h))
                cv2.line(frame, pt1, pt2, (200, 200, 0), 1, cv2.LINE_AA)

            # Lips outer: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185
            lips_outer = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
            for i in range(len(lips_outer)):
                idx1, idx2 = lips_outer[i], lips_outer[(i + 1) % len(lips_outer)]
                pt1 = (int(landmarks[idx1].x * w), int(landmarks[idx1].y * h))
                pt2 = (int(landmarks[idx2].x * w), int(landmarks[idx2].y * h))
                cv2.line(frame, pt1, pt2, (0, 150, 255), 1, cv2.LINE_AA)

        except (AttributeError, IndexError, TypeError):
            # Not MediaPipe landmarks or invalid data
            pass

    def run(self):
        """Main application loop."""
        self.logger.log(f"Starting camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_cfg.video_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_cfg.video_size[1])

        cv2.namedWindow(self.display_cfg.window_name)
        self.logger.log("Press 'g' gaze, 'z' auto-calibrate, 'c' calibration (WASD), 'v' vectors, 'l' iris, 'm' mesh, 'f' FPS, 'q' quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.log("Failed to read frame")
                break

            # Calculate FPS
            self.frame_count += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_start_time = time.time()

            if self.gaze_correction_enabled:
                display_frame = self.process_frame(frame)
            else:
                display_frame = frame.copy()

            self.draw_status(display_frame)

            # Draw calibration overlay if enabled
            if self.calibration_mode:
                self.draw_calibration_overlay(display_frame)

            cv2.imshow(self.display_cfg.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("g"):
                self.gaze_correction_enabled = not self.gaze_correction_enabled
                self.logger.log(
                    f"Gaze correction: {'enabled' if self.gaze_correction_enabled else 'disabled'}"
                )
            elif key == ord("c"):
                self.calibration_mode = not self.calibration_mode
                self.logger.log(
                    f"Calibration mode: {'enabled' if self.calibration_mode else 'disabled'}"
                )
            elif key == ord("v"):
                self.vector_visualization_enabled = not self.vector_visualization_enabled
                self.gaze_corrector.set_vector_visualization(self.vector_visualization_enabled)
                self.logger.log(
                    f"Vector visualization: {'enabled' if self.vector_visualization_enabled else 'disabled'}"
                )
            elif key == ord("f"):
                self.fps_display_enabled = not self.fps_display_enabled
                self.logger.log(
                    f"FPS display: {'enabled' if self.fps_display_enabled else 'disabled'}"
                )
            elif key == ord("l"):
                self.landmark_visualization_enabled = not self.landmark_visualization_enabled
                self.logger.log(
                    f"Iris landmark visualization: {'enabled' if self.landmark_visualization_enabled else 'disabled'}"
                )
            elif key == ord("m"):
                self.face_mesh_visualization_enabled = not self.face_mesh_visualization_enabled
                self.logger.log(
                    f"Face mesh visualization: {'enabled' if self.face_mesh_visualization_enabled else 'disabled'}"
                )
            elif key == ord("z"):
                # Auto-calibrate works independently of calibration mode
                self.gaze_corrector.auto_calibrate()
            elif self.calibration_mode:
                self.handle_calibration_key(key)

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.gaze_corrector.close()
        self.logger.log("Shutdown complete")
