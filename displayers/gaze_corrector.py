"""
Gaze Corrector Module

This module provides the gaze correction model wrapper and inference logic,
decoupled from face detection and eye extraction.
"""

import math
import numpy as np
import tensorflow as tf
import cv2
from dataclasses import dataclass
from typing import Optional

import tf_models.flx as flx_model
from utils.config import get_config
from utils.logger import Logger
from displayers.face_predictor import EyeData, FaceData


################################################################################
# Configuration
################################################################################


@dataclass
class GazeModelConfig:
    """Configuration for the gaze correction model."""

    model_dir: str = "./weights/warping_model/flx/12/"
    eye_input_size: tuple[int, int] = (48, 64)  # (height, width)
    ef_dim: int = 12
    channel: int = 3


@dataclass
class CameraConfig:
    """Camera and screen geometry configuration."""

    focal_length: float = 650.0
    ipd: float = 6.3  # Inter-pupillary distance in cm
    camera_offset: tuple[float, float, float] = (0, -21, -1)  # relative to screen center
    video_size: tuple[int, int] = (640, 480)

    def copy(self) -> "CameraConfig":
        """Create a copy of this config."""
        return CameraConfig(
            focal_length=self.focal_length,
            ipd=self.ipd,
            camera_offset=self.camera_offset,
            video_size=self.video_size,
        )


################################################################################
# Gaze Correction Model
################################################################################


class GazeModel:
    """TensorFlow model wrapper for eye gaze correction."""

    def __init__(self, config: Optional[GazeModelConfig] = None):
        self.cfg = config or GazeModelConfig()
        self.logger = Logger("GazeModel")
        self._load_models()

    def _load_models(self):
        """Load left and right eye models."""
        general_cfg, _ = get_config()
        model_cfg = flx_model.ModelConfig.parse_from(general_cfg)

        # Left eye model
        self.logger.log("Loading left eye model...")
        with tf.Graph().as_default() as g_left:
            with tf.name_scope("inputs"):
                self.le_img = tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], self.cfg.channel],
                )
                self.le_fp = tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], self.cfg.ef_dim],
                )
                self.le_ang = tf.compat.v1.placeholder(tf.float32, [None, 2])

            self.le_pred, _, _ = flx_model.inference(
                self.le_img, self.le_fp, self.le_ang, False, model_cfg
            )
            self.l_sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(allow_soft_placement=True),
                graph=g_left,
            )
            self._restore_checkpoint(self.l_sess, self.cfg.model_dir + "L/")

        # Right eye model
        self.logger.log("Loading right eye model...")
        with tf.Graph().as_default() as g_right:
            with tf.name_scope("inputs"):
                self.re_img = tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], self.cfg.channel],
                )
                self.re_fp = tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], self.cfg.ef_dim],
                )
                self.re_ang = tf.compat.v1.placeholder(tf.float32, [None, 2])

            self.re_pred, _, _ = flx_model.inference(
                self.re_img, self.re_fp, self.re_ang, False, model_cfg
            )
            self.r_sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(allow_soft_placement=True),
                graph=g_right,
            )
            self._restore_checkpoint(self.r_sess, self.cfg.model_dir + "R/")

        self.logger.log("Models loaded successfully")

    def _restore_checkpoint(self, sess, model_dir: str):
        """Restore model from checkpoint."""
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        ckpt = tf.compat.v1.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            self.logger.log(f"Warning: No checkpoint found in {model_dir}")

    def infer_eye(
        self, eye: str, img: np.ndarray, anchor_map: np.ndarray, angle: list
    ) -> np.ndarray:
        """
        Run inference for a single eye.

        Args:
            eye: "L" or "R"
            img: Eye image normalized to [0, 1], shape (H, W, 3)
            anchor_map: Feature point map, shape (H, W, ef_dim)
            angle: [vertical, horizontal] correction angles

        Returns:
            Corrected eye image, shape (H, W, 3)
        """
        if eye == "L":
            result = self.l_sess.run(
                self.le_pred,
                feed_dict={
                    self.le_img: np.expand_dims(img, axis=0),
                    self.le_fp: np.expand_dims(anchor_map, axis=0),
                    self.le_ang: np.expand_dims(angle, axis=0),
                },
            )
        else:
            result = self.r_sess.run(
                self.re_pred,
                feed_dict={
                    self.re_img: np.expand_dims(img, axis=0),
                    self.re_fp: np.expand_dims(anchor_map, axis=0),
                    self.re_ang: np.expand_dims(angle, axis=0),
                },
            )
        return result.reshape(self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], 3)

    def close(self):
        """Close TensorFlow sessions."""
        self.l_sess.close()
        self.r_sess.close()


################################################################################
# Gaze Corrector
################################################################################


class GazeCorrector:
    """
    High-level gaze correction interface.
    
    Takes FaceData from a FacePredictor and applies gaze correction.
    """

    def __init__(
        self,
        model_config: Optional[GazeModelConfig] = None,
        camera_config: Optional[CameraConfig] = None,
    ):
        self.model_cfg = model_config or GazeModelConfig()
        self.camera_cfg = camera_config or CameraConfig()
        self.logger = Logger("GazeCorrector")

        self.model = GazeModel(self.model_cfg)

        # Pixel border to cut when replacing eyes (reduces edge artifacts)
        self.pixel_cut = (3, 4)

        # Last estimated eye position (for visualization)
        self.last_eye_position: list[float] = [0, 0, -60]
        
        # Vector visualization
        self.draw_vectors_enabled: bool = False
        self.last_gaze_direction: Optional[np.ndarray] = None
        self.last_eye_to_camera: Optional[np.ndarray] = None
        self.last_eye_center_pixels: Optional[tuple[float, float]] = None

    ############################################################################
    # Camera Offset Adjustment API
    ############################################################################

    def get_camera_offset(self) -> tuple[float, float, float]:
        """Get current camera offset (x, y, z) in cm."""
        return self.camera_cfg.camera_offset

    def set_camera_offset(self, x: float, y: float, z: float) -> None:
        """
        Set camera offset relative to screen center.

        Args:
            x: Horizontal offset in cm (positive = right)
            y: Vertical offset in cm (positive = down)
            z: Depth offset in cm (negative = behind screen)
        """
        self.camera_cfg.camera_offset = (x, y, z)
        self.logger.log(f"Camera offset set to: ({x:.1f}, {y:.1f}, {z:.1f})")

    def adjust_camera_offset(self, dx: float = 0, dy: float = 0, dz: float = 0) -> tuple[float, float, float]:
        """
        Adjust camera offset by delta values.

        Args:
            dx: Change in X (horizontal)
            dy: Change in Y (vertical)
            dz: Change in Z (depth)

        Returns:
            New camera offset tuple
        """
        x, y, z = self.camera_cfg.camera_offset
        self.camera_cfg.camera_offset = (x + dx, y + dy, z + dz)
        return self.camera_cfg.camera_offset

    def get_last_eye_position(self) -> list[float]:
        """Get the last estimated eye position [x, y, z] in cm."""
        return self.last_eye_position

    ############################################################################
    # Vector Visualization API
    ############################################################################

    def toggle_vector_visualization(self) -> bool:
        """Toggle vector visualization on/off. Returns new state."""
        self.draw_vectors_enabled = not self.draw_vectors_enabled
        return self.draw_vectors_enabled

    def set_vector_visualization(self, enabled: bool) -> None:
        """Set vector visualization state."""
        self.draw_vectors_enabled = enabled

    def is_vector_visualization_enabled(self) -> bool:
        """Check if vector visualization is enabled."""
        return self.draw_vectors_enabled

    ############################################################################
    # Focal Length Adjustment API
    ############################################################################

    def get_focal_length(self) -> float:
        """Get current focal length in pixels."""
        return self.camera_cfg.focal_length

    def set_focal_length(self, focal_length: float) -> None:
        """
        Set focal length.

        Args:
            focal_length: Focal length in pixels (typically 500-1000)
        """
        self.camera_cfg.focal_length = focal_length
        self.logger.log(f"Focal length set to: {focal_length:.1f}")

    def adjust_focal_length(self, delta: float) -> float:
        """
        Adjust focal length by delta value.

        Args:
            delta: Change in focal length (pixels)

        Returns:
            New focal length
        """
        self.camera_cfg.focal_length += delta
        return self.camera_cfg.focal_length

    # TODO: (v2) Compute the normal vector of eye plane and the vector from eye to camera. Then, compute angle between them.
    def estimate_gaze_angle(
        self, le_center: tuple[float, float], re_center: tuple[float, float]
    ) -> tuple[list[int], list[float]]:
        """
        Estimate gaze redirection angles based on eye positions.

        Args:
            le_center: Left eye center (x, y) in pixels
            re_center: Right eye center (x, y) in pixels

        Returns:
            (alpha [vertical, horizontal], eye_position [x, y, z])
        """
        cfg = self.camera_cfg

        # Estimate eye depth from inter-pupillary distance
        ipd_pixels = np.sqrt(
            (le_center[0] - re_center[0]) ** 2 + (le_center[1] - re_center[1]) ** 2
        )
        eye_z = -(cfg.focal_length * cfg.ipd) / ipd_pixels

        # Estimate eye position in 3D (camera coordinates, cm)
        eye_x = (
            -abs(eye_z)
            * (le_center[0] + re_center[0] - cfg.video_size[0])
            / (2 * cfg.focal_length)
            + cfg.camera_offset[0]
        )
        eye_y = (
            abs(eye_z)
            * (le_center[1] + re_center[1] - cfg.video_size[1])
            / (2 * cfg.focal_length)
            + cfg.camera_offset[1]
        )

        eye_position = [eye_x, eye_y, eye_z]

        # Store for visualization
        self.last_eye_position = eye_position

        # Target gaze point (looking at camera)
        target = (0, 0, 0)

        # Calculate angles
        a_v = math.degrees(math.atan((target[1] - eye_y) / (target[2] - eye_z)))
        a_h = math.degrees(math.atan((target[0] - eye_x) / (target[2] - eye_z)))

        # Add camera offset angles
        a_v += math.degrees(
            math.atan((eye_y - cfg.camera_offset[1]) / (cfg.camera_offset[2] - eye_z))
        )
        a_h += math.degrees(
            math.atan((eye_x - cfg.camera_offset[0]) / (cfg.camera_offset[2] - eye_z))
        )

        return [int(a_v), int(a_h)], eye_position

    def estimate_gaze_angle_v2(
        self,
        le_center: tuple[float, float],
        re_center: tuple[float, float],
        le_iris_points: Optional[list[tuple[float, float, float]]],
        re_iris_points: Optional[list[tuple[float, float, float]]],
    ) -> tuple[list[int], list[float]]:
        """
        Estimate gaze redirection angles using eye plane normal vectors.

        This method computes the angle between the current gaze direction
        (normal vector of the eye plane) and the eye-to-camera vector.

        Args:
            le_center: Left eye center (x, y) in pixels
            re_center: Right eye center (x, y) in pixels
            le_iris_points: Left eye iris points (4 points with x, y, z), or None
            re_iris_points: Right eye iris points (4 points with x, y, z), or None

        Returns:
            (alpha [vertical, horizontal], eye_position [x, y, z])
        """
        cfg = self.camera_cfg

        # Estimate eye depth from inter-pupillary distance
        ipd_pixels = np.sqrt(
            (le_center[0] - re_center[0]) ** 2 + (le_center[1] - re_center[1]) ** 2
        )
        eye_z = -(cfg.focal_length * cfg.ipd) / ipd_pixels

        # Estimate eye position in 3D (camera coordinates, cm)
        eye_x = (
            -abs(eye_z)
            * (le_center[0] + re_center[0] - cfg.video_size[0])
            / (2 * cfg.focal_length)
            + cfg.camera_offset[0]
        )
        eye_y = (
            abs(eye_z)
            * (le_center[1] + re_center[1] - cfg.video_size[1])
            / (2 * cfg.focal_length)
            + cfg.camera_offset[1]
        )

        eye_position = [eye_x, eye_y, eye_z]

        # Store for visualization
        self.last_eye_position = eye_position

        # Step 1: Convert all iris points from both eyes to camera coordinates
        center_iris_points = []
        
        for i in range(4):
            center_ipd_pixels = np.sqrt(
                (le_iris_points[i][0] - re_iris_points[i][0]) ** 2 +
                (le_iris_points[i][1] - re_iris_points[i][1]) ** 2
            )
            center_eye_z = -(cfg.focal_length * cfg.ipd) / center_ipd_pixels
            center_eye_x = (
                -abs(center_eye_z)
                * (le_iris_points[i][0] + re_iris_points[i][0] - cfg.video_size[0])
                / (2 * cfg.focal_length)
                + cfg.camera_offset[0]
            )
            center_eye_y = (
                abs(center_eye_z)
                * (le_iris_points[i][1] + re_iris_points[i][1] - cfg.video_size[1])
                / (2 * cfg.focal_length)
                + cfg.camera_offset[1]
            )
            center_iris_points.append((center_eye_x, center_eye_y, center_eye_z))
        
        # Step 3: Compute gaze direction (normal vector) using iris points
        p0 = np.array(center_iris_points[0])  # Top
        p1 = np.array(center_iris_points[1])  # Right
        p2 = np.array(center_iris_points[2])  # Bottom
        p3 = np.array(center_iris_points[3])  # Left
        
        # Create two vectors on the iris plane
        v1 = p2 - p0  # Vector from one corner to opposite corner
        v2 = p1 - p3  # Vector from another corner to opposite corner
        
        # Cross product gives the normal vector (gaze direction)
        gaze_direction = np.cross(v1, v2)
        
        # Normalize the gaze direction vector
        gaze_norm = np.linalg.norm(gaze_direction)
        if gaze_norm < 1e-6:
            # Degenerate case: vectors are parallel or zero
            return self.estimate_gaze_angle(le_center, re_center)
        gaze_direction = gaze_direction / gaze_norm
        
        # Ensure the normal points towards the camera (negative z direction in camera space)
        # In camera coordinates, objects in front have negative z
        if gaze_direction[2] > 0:
            gaze_direction = -gaze_direction

        # Step 4: Compute eye-to-camera vector
        # In camera coordinates, the camera is at the origin
        camera_pos = np.array([0, 0, 0])
        # Use center_iris position instead of eye_position for more accurate gaze calculation
        iris_to_camera = camera_pos - np.array(eye_position)

        # Normalize iris-to-camera vector
        itc_norm = np.linalg.norm(iris_to_camera)
        if itc_norm < 1e-6:
            return self.estimate_gaze_angle(le_center, re_center)
        iris_to_camera = iris_to_camera / itc_norm
        
        # Store vectors for visualization
        self.last_gaze_direction = gaze_direction
        self.last_eye_to_camera = iris_to_camera
        self.last_eye_center_pixels = (
            (le_center[0] + re_center[0]) / 2,
            (le_center[1] + re_center[1]) / 2
        )

        # Step 5: Compute angles between gaze direction and iris-to-camera vector
        # Project onto XZ plane (horizontal) and YZ plane (vertical)

        # Horizontal angle (rotation around Y axis)
        gaze_xz = np.array([gaze_direction[0], 0, gaze_direction[2]])
        itc_xz = np.array([iris_to_camera[0], 0, iris_to_camera[2]])
        gaze_xz_norm = np.linalg.norm(gaze_xz)
        itc_xz_norm = np.linalg.norm(itc_xz)

        if gaze_xz_norm > 1e-6 and itc_xz_norm > 1e-6:
            gaze_xz = gaze_xz / gaze_xz_norm
            itc_xz = itc_xz / itc_xz_norm
            cos_h = np.clip(np.dot(gaze_xz, itc_xz), -1.0, 1.0)
            a_h = math.degrees(math.acos(cos_h))
            # Determine sign using cross product
            cross_h = np.cross(gaze_xz, itc_xz)
            if cross_h[1] < 0:
                a_h = -a_h
        else:
            a_h = 0

        # Vertical angle (rotation around X axis)
        gaze_yz = np.array([0, gaze_direction[1], gaze_direction[2]])
        itc_yz = np.array([0, iris_to_camera[1], iris_to_camera[2]])
        gaze_yz_norm = np.linalg.norm(gaze_yz)
        itc_yz_norm = np.linalg.norm(itc_yz)

        if gaze_yz_norm > 1e-6 and itc_yz_norm > 1e-6:
            gaze_yz = gaze_yz / gaze_yz_norm
            itc_yz = itc_yz / itc_yz_norm
            cos_v = np.clip(np.dot(gaze_yz, itc_yz), -1.0, 1.0)
            a_v = math.degrees(math.acos(cos_v))
            # Determine sign using cross product
            cross_v = np.cross(gaze_yz, itc_yz)
            if cross_v[0] < 0:
                a_v = -a_v
        else:
            a_v = 0

        return [int(a_v), int(a_h)], eye_position

    def correct_eye(
        self, eye_data: EyeData, eye_side: str, angle: list[int]
    ) -> np.ndarray:
        """
        Apply gaze correction to a single eye.

        Args:
            eye_data: Eye extraction data
            eye_side: "L" or "R"
            angle: [vertical, horizontal] correction angles

        Returns:
            Corrected eye image resized to original size
        """
        result = self.model.infer_eye(
            eye_side, eye_data.image, eye_data.anchor_map, angle
        )
        # Resize back to original size
        return cv2.resize(result, (eye_data.original_size[1], eye_data.original_size[0]))

    def apply_correction(self, frame: np.ndarray, face_data: FaceData) -> np.ndarray:
        """
        Apply gaze correction to a frame using extracted face data.

        Args:
            frame: BGR video frame to modify
            face_data: Extracted face/eye data from FacePredictor

        Returns:
            Frame with corrected gaze
        """
        if face_data.left_eye is None or face_data.right_eye is None:
            return frame

        le = face_data.left_eye
        re = face_data.right_eye
        
        # Store eye center for visualization
        self.last_eye_center_pixels = (
            (le.center[0] + re.center[0]) / 2,
            (le.center[1] + re.center[1]) / 2
        )

        # Estimate gaze angle using v2 if iris_points are available, otherwise fallback to v1
        if le.iris_points is not None or re.iris_points is not None:
            alpha, _ = self.estimate_gaze_angle_v2(
                le.center, re.center, le.iris_points, re.iris_points
            )
        else:
            alpha, _ = self.estimate_gaze_angle(le.center, re.center)

        # Correct both eyes
        le_corrected = self.correct_eye(le, "L", alpha)
        re_corrected = self.correct_eye(re, "R", alpha)

        # Replace eye regions in frame (with border cropping)
        pc = self.pixel_cut
        frame[
            le.top_left[0] + pc[0] : le.top_left[0] + le.original_size[0] - pc[0],
            le.top_left[1] + pc[1] : le.top_left[1] + le.original_size[1] - pc[1],
        ] = (le_corrected[pc[0] : -pc[0], pc[1] : -pc[1]] * 255)

        frame[
            re.top_left[0] + pc[0] : re.top_left[0] + re.original_size[0] - pc[0],
            re.top_left[1] + pc[1] : re.top_left[1] + re.original_size[1] - pc[1],
        ] = (re_corrected[pc[0] : -pc[0], pc[1] : -pc[1]] * 255)

        return frame

    def draw_vectors(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw gaze direction and eye-to-camera vectors on the frame.
        
        Args:
            frame: BGR video frame to draw on
            
        Returns:
            Frame with vectors drawn
        """
        if not self.draw_vectors_enabled:
            return frame
            
        if self.last_eye_center_pixels is None:
            return frame
        
        center_x = int(self.last_eye_center_pixels[0])
        center_y = int(self.last_eye_center_pixels[1])
        
        # Draw eye center point
        cv2.circle(frame, (center_x, center_y), 5, (255, 255, 0), -1)
        
        # Scale factor for vector visualization (longer vectors for visibility)
        vector_scale = 80
        
        if self.last_gaze_direction is not None:
            # Draw gaze direction vector (green)
            gaze_end_x = int(center_x + self.last_gaze_direction[0] * vector_scale)
            gaze_end_y = int(center_y + self.last_gaze_direction[1] * vector_scale)
            cv2.arrowedLine(
                frame,
                (center_x, center_y),
                (gaze_end_x, gaze_end_y),
                (0, 255, 0),  # Green
                3,
                tipLength=0.3
            )
            # Label
            cv2.putText(
                frame,
                "Gaze",
                (gaze_end_x + 5, gaze_end_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        
        if self.last_eye_to_camera is not None:
            # Draw eye-to-camera vector (cyan)
            cam_end_x = int(center_x + self.last_eye_to_camera[0] * vector_scale)
            cam_end_y = int(center_y + self.last_eye_to_camera[1] * vector_scale)
            cv2.arrowedLine(
                frame,
                (center_x, center_y),
                (cam_end_x, cam_end_y),
                (255, 255, 0),  # Cyan
                3,
                tipLength=0.3
            )
            # Label
            cv2.putText(
                frame,
                "To-Cam",
                (cam_end_x + 5, cam_end_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )
        
        return frame

    def close(self):
        """Release model resources."""
        self.model.close()
