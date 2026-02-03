"""
Gaze Corrector V1 Module

This module provides the gaze correction model wrapper and inference logic,
with YAML-based configuration and database-backed user settings.
"""

import math
import yaml
import numpy as np
import tensorflow as tf
import cv2
from dataclasses import dataclass
from typing import Optional

from tf_models.gaze_corrector_v1 import gaze_warp_model
from utils.logger import Logger
from model_managers.user_settings_db import UserSettingsDB


################################################################################
# Configuration Classes
################################################################################


@dataclass
class GazeWarpModelConfig:
    """Hyperparameters for the gaze warp model."""
    height: int = 48
    width: int = 64
    encoded_angle_dim: int = 16


@dataclass
class GazeModelConfig:
    """Configuration for the gaze correction model."""
    
    model_dir: str = "./weights/warping_model/flx/12/"
    eye_input_size: tuple[int, int] = (48, 64)  # (height, width)
    ef_dim: int = 12
    channel: int = 3
    gaze_warp_model: GazeWarpModelConfig = None
    
    def __post_init__(self):
        if self.gaze_warp_model is None:
            self.gaze_warp_model = GazeWarpModelConfig()
        elif isinstance(self.gaze_warp_model, dict):
            self.gaze_warp_model = GazeWarpModelConfig(**self.gaze_warp_model)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GazeModelConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert eye_input_size from list to tuple
        if 'eye_input_size' in data and isinstance(data['eye_input_size'], list):
            data['eye_input_size'] = tuple(data['eye_input_size'])
        
        return cls(**data)


@dataclass
class CameraUserSetting:
    """User-adjustable camera and screen geometry settings."""

    focal_length: float = 650.0
    ipd: float = 6.3  # Inter-pupillary distance in cm
    camera_offset: tuple[float, float, float] = (0, 5, -1)  # relative to screen center (Y positive = above)
    # Manual gaze offset for fine-tuning (degrees) - adjust with WASD while looking at camera
    gaze_offset_v: float = 0.0  # Vertical offset (positive = correct upward)
    gaze_offset_h: float = 0.0  # Horizontal offset (positive = correct rightward)
    # Gaze scale factor - how many degrees per 1/4 eye width of iris movement (default 20)
    gaze_scale: float = 20.0
    # Neutral gaze direction (iris plane normal when looking at camera) - set by auto-calibrate
    # Stored as (nx, ny, nz) unit vector. None means not calibrated.
    neutral_gaze: Optional[tuple[float, float, float]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            'focal_length': self.focal_length,
            'ipd': self.ipd,
            'camera_offset': list(self.camera_offset),
            'gaze_offset_v': self.gaze_offset_v,
            'gaze_offset_h': self.gaze_offset_h,
            'gaze_scale': self.gaze_scale,
            'neutral_gaze': list(self.neutral_gaze) if self.neutral_gaze else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraUserSetting":
        """Load from dictionary."""
        if 'camera_offset' in data and isinstance(data['camera_offset'], list):
            data['camera_offset'] = tuple(data['camera_offset'])
        if 'neutral_gaze' in data and isinstance(data['neutral_gaze'], list):
            data['neutral_gaze'] = tuple(data['neutral_gaze'])
        return cls(**data)


################################################################################
# Gaze Correction Model
################################################################################


class GazeModel:
    """TensorFlow model wrapper for eye gaze correction."""

    def __init__(self, config: GazeModelConfig):
        self.cfg = config
        self.logger = Logger("GazeModel")
        self._load_models()

    def _load_models(self):
        """Load left and right eye models."""
        # Build ModelConfig for gaze_warp_model
        model_cfg = gaze_warp_model.ModelConfig(
            height=self.cfg.gaze_warp_model.height,
            width=self.cfg.gaze_warp_model.width,
            encoded_angle_dim=self.cfg.gaze_warp_model.encoded_angle_dim,
        )

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

            self.le_pred, _, _ = gaze_warp_model.build_inference_graph(
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

            self.re_pred, _, _ = gaze_warp_model.build_inference_graph(
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
    High-level gaze correction interface with database-backed user settings.
    
    Takes FaceData from a FacePredictor and applies gaze correction.
    video_size is passed from outside via apply_correction.
    """

    def __init__(
        self,
        config_path: str = "./model_managers/gaze_corrector_v1_01.yaml",
        db_path: str = "./user_settings.db",
        setting_name: str = "camera_default",
        verbose: bool = False,
    ):
        """
        Initialize gaze corrector.
        
        Args:
            config_path: Path to YAML configuration file
            db_path: Path to SQLite database
            setting_name: Name of camera setting to load from database
        """
        self.logger = Logger("GazeCorrector")
        
        # Load model configuration from YAML
        self.model_cfg = GazeModelConfig.from_yaml(config_path)
        self.logger.log(f"Loaded model config from: {config_path}")
        
        # Initialize database
        self.db = UserSettingsDB(db_path)
        self.setting_name = setting_name
        
        # Load camera settings from database or use defaults
        self.camera_settings = self._load_camera_settings()
        
        # Initialize model
        self.model = GazeModel(self.model_cfg)

        # Pixel border to cut when replacing eyes (reduces edge artifacts)
        # The warping model produces dark edge artifacts - cut aggressively
        self.pixel_cut = (8, 12)  # (rows, cols) to cut from each edge
        self.blend_border = 12  # Additional pixels for alpha blending

        # Last estimated eye position (for visualization)
        self.last_eye_position: list[float] = [0, 0, -60]
        
        # Vector visualization
        self.draw_vectors_enabled: bool = False
        self.last_gaze_direction: Optional[np.ndarray] = None
        self.last_eye_to_camera: Optional[np.ndarray] = None
        self.last_eye_center_pixels: Optional[tuple[float, float]] = None

        # Current iris plane normal (for compatibility, stores normalized displacement)
        self.current_iris_normal: Optional[np.ndarray] = None

        # Raw gaze angles BEFORE calibration offset (used for calibration capture)
        self.last_raw_gaze: tuple[float, float] = (0.0, 0.0)  # (vertical, horizontal) in degrees

        # Last detected gaze AFTER calibration offset (for display)
        self.last_detected_gaze: tuple[float, float] = (0.0, 0.0)  # (vertical, horizontal) in degrees

        # Last head pose for display (pitch, yaw, roll in degrees)
        self.last_head_pose: Optional[tuple[float, float, float]] = None

        # Debug mode for v2 gaze estimation
        self.debug_v2: bool = verbose

    def _load_camera_settings(self) -> CameraUserSetting:
        """Load camera settings from database or return defaults."""
        saved = self.db.get_setting(self.setting_name)
        if saved:
            self.logger.log(f"Loaded camera settings from database: {self.setting_name}")
            return CameraUserSetting.from_dict(saved)
        else:
            self.logger.log("Using default camera settings")
            settings = CameraUserSetting()
            # Save defaults to database
            self.db.save_setting(self.setting_name, settings.to_dict())
            return settings

    def save_camera_settings(self):
        """Save current camera settings to database."""
        self.db.save_setting(self.setting_name, self.camera_settings.to_dict())
        self.logger.log(f"Saved camera settings to database: {self.setting_name}")

    ############################################################################
    # Camera Offset Adjustment API
    ############################################################################

    def get_camera_offset(self) -> tuple[float, float, float]:
        """Get current camera offset (x, y, z) in cm."""
        return self.camera_settings.camera_offset

    def set_camera_offset(self, x: float, y: float, z: float) -> None:
        """
        Set camera offset relative to screen center.

        Args:
            x: Horizontal offset in cm (positive = right)
            y: Vertical offset in cm (positive = down)
            z: Depth offset in cm (negative = behind screen)
        """
        self.camera_settings.camera_offset = (x, y, z)
        self.save_camera_settings()
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
        x, y, z = self.camera_settings.camera_offset
        self.camera_settings.camera_offset = (x + dx, y + dy, z + dz)
        self.save_camera_settings()
        return self.camera_settings.camera_offset

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
        return self.camera_settings.focal_length

    def set_focal_length(self, focal_length: float) -> None:
        """
        Set focal length.

        Args:
            focal_length: Focal length in pixels (typically 500-1000)
        """
        self.camera_settings.focal_length = focal_length
        self.save_camera_settings()
        self.logger.log(f"Focal length set to: {focal_length:.1f}")

    def adjust_focal_length(self, delta: float) -> float:
        """
        Adjust focal length by delta value.

        Args:
            delta: Change in focal length (pixels)

        Returns:
            New focal length
        """
        self.camera_settings.focal_length += delta
        self.save_camera_settings()
        return self.camera_settings.focal_length

    ############################################################################
    # IPD Adjustment API
    ############################################################################

    def get_ipd(self) -> float:
        """Get inter-pupillary distance in cm."""
        return self.camera_settings.ipd

    def set_ipd(self, ipd: float) -> None:
        """
        Set inter-pupillary distance.

        Args:
            ipd: IPD in cm (typically 5.5-7.0)
        """
        self.camera_settings.ipd = ipd
        self.save_camera_settings()
        self.logger.log(f"IPD set to: {ipd:.1f} cm")

    ############################################################################
    # Manual Gaze Offset API (WASD adjustment)
    ############################################################################

    def get_gaze_offset(self) -> tuple[float, float]:
        """Get manual gaze offset (vertical, horizontal) in degrees."""
        return (self.camera_settings.gaze_offset_v, self.camera_settings.gaze_offset_h)

    def set_gaze_offset(self, v: float, h: float) -> None:
        """
        Set manual gaze offset.

        Args:
            v: Vertical offset in degrees (positive = correct upward)
            h: Horizontal offset in degrees (positive = correct rightward)
        """
        self.camera_settings.gaze_offset_v = v
        self.camera_settings.gaze_offset_h = h
        self.save_camera_settings()
        self.logger.log(f"Gaze offset set to: V={v:+.1f}°, H={h:+.1f}°")

    def adjust_gaze_offset(self, dv: float = 0, dh: float = 0) -> tuple[float, float]:
        """
        Adjust manual gaze offset by delta values.

        Args:
            dv: Change in vertical offset (degrees)
            dh: Change in horizontal offset (degrees)

        Returns:
            New gaze offset tuple (v, h)
        """
        self.camera_settings.gaze_offset_v += dv
        self.camera_settings.gaze_offset_h += dh
        self.save_camera_settings()
        return (self.camera_settings.gaze_offset_v, self.camera_settings.gaze_offset_h)

    def reset_gaze_offset(self) -> None:
        """Reset manual gaze offset to zero."""
        self.camera_settings.gaze_offset_v = 0.0
        self.camera_settings.gaze_offset_h = 0.0
        self.save_camera_settings()
        self.logger.log("Gaze offset reset to zero")

    def reset_neutral_gaze(self) -> None:
        """Reset the calibrated neutral gaze direction."""
        self.camera_settings.neutral_gaze = None
        self.current_iris_normal = None
        self.save_camera_settings()
        self.logger.log("Neutral gaze calibration reset")

    def is_calibrated(self) -> bool:
        """Check if gaze is calibrated (neutral_gaze is set)."""
        return self.camera_settings.neutral_gaze is not None

    def auto_calibrate(self) -> tuple[float, float]:
        """
        Auto-calibrate gaze by capturing the current raw gaze as "neutral".

        Call this while looking directly at the camera. The system will:
        1. Capture the current raw gaze angles as the "neutral" gaze direction
        2. Reset manual offset to zero (calibration handles everything)

        All future gaze corrections will be relative to this calibrated neutral
        direction. If you're looking 5° up when you press 'z', that becomes the
        new "center" and all corrections are measured from there.

        Returns:
            New gaze offset tuple (v, h) - will be (0, 0) after calibration
        """
        # Capture current raw gaze as the neutral position
        raw_v, raw_h = self.last_raw_gaze

        # Store as (v, h, 0) tuple for compatibility with settings storage
        # Only v and h are used; third component is ignored
        self.camera_settings.neutral_gaze = (raw_v, raw_h, 0.0)

        # Reset manual offset since calibration handles the baseline
        self.camera_settings.gaze_offset_v = 0.0
        self.camera_settings.gaze_offset_h = 0.0
        self.save_camera_settings()

        self.logger.log(
            f"Auto-calibrated: neutral_gaze=({raw_v:.1f}°, {raw_h:.1f}°), "
            f"manual offset reset to (0°, 0°)"
        )

        return (self.camera_settings.gaze_offset_v, self.camera_settings.gaze_offset_h)

    ############################################################################
    # Gaze Scale Adjustment API
    ############################################################################

    def get_gaze_scale(self) -> float:
        """Get gaze scale factor (degrees per 1/4 eye width of iris movement)."""
        return self.camera_settings.gaze_scale

    def set_gaze_scale(self, scale: float) -> None:
        """
        Set gaze scale factor.

        Args:
            scale: Degrees of gaze shift per 1/4 eye width of iris movement.
                   Higher = more sensitive correction. Default is 20.0.
        """
        self.camera_settings.gaze_scale = max(5.0, min(50.0, scale))  # Clamp to reasonable range
        self.save_camera_settings()
        self.logger.log(f"Gaze scale set to: {self.camera_settings.gaze_scale:.1f}")

    def adjust_gaze_scale(self, delta: float) -> float:
        """
        Adjust gaze scale by delta value.

        Args:
            delta: Change in gaze scale

        Returns:
            New gaze scale
        """
        new_scale = max(5.0, min(50.0, self.camera_settings.gaze_scale + delta))
        self.camera_settings.gaze_scale = new_scale
        self.save_camera_settings()
        return new_scale

    ############################################################################
    # Gaze Estimation and Correction
    ############################################################################

    # TODO: (v2) Compute the normal vector of eye plane and the vector from eye to camera. Then, compute angle between them.
    def estimate_gaze_angle(
        self, 
        le_center: tuple[float, float], 
        re_center: tuple[float, float],
        video_size: tuple[int, int],
    ) -> tuple[list[int], list[float]]:
        """
        Estimate gaze redirection angles based on eye positions.

        Args:
            le_center: Left eye center (x, y) in pixels
            re_center: Right eye center (x, y) in pixels
            video_size: (width, height) of video frame

        Returns:
            (alpha [vertical, horizontal], eye_position [x, y, z])
        """
        settings = self.camera_settings

        # Estimate eye depth from inter-pupillary distance
        ipd_pixels = np.sqrt(
            (le_center[0] - re_center[0]) ** 2 + (le_center[1] - re_center[1]) ** 2
        )
        eye_z = -(settings.focal_length * settings.ipd) / ipd_pixels

        # Estimate eye position in 3D (camera coordinates, cm)
        eye_x = (
            -abs(eye_z)
            * (le_center[0] + re_center[0] - video_size[0])
            / (2 * settings.focal_length)
            + settings.camera_offset[0]
        )
        eye_y = (
            abs(eye_z)
            * (le_center[1] + re_center[1] - video_size[1])
            / (2 * settings.focal_length)
            + settings.camera_offset[1]
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
            math.atan((eye_y - settings.camera_offset[1]) / (settings.camera_offset[2] - eye_z))
        )
        a_h += math.degrees(
            math.atan((eye_x - settings.camera_offset[0]) / (settings.camera_offset[2] - eye_z))
        )

        # Store for display (v1 doesn't have calibration, just uses geometry)
        self.last_raw_gaze = (a_v, a_h)
        self.last_detected_gaze = (a_v, a_h)

        # Store eye center for visualization
        self.last_eye_center_pixels = (
            (le_center[0] + re_center[0]) / 2,
            (le_center[1] + re_center[1]) / 2
        )

        # Compute visualization vectors
        a_v_rad = math.radians(a_v)
        a_h_rad = math.radians(a_h)
        self.last_gaze_direction = np.array([
            math.sin(a_h_rad),
            math.sin(a_v_rad),
            math.cos(a_v_rad) * math.cos(a_h_rad)
        ])
        self.last_eye_to_camera = np.array([0, 0, 1])

        if self.debug_v2:
            self.logger.log(
                f"v1: eye_pos=({eye_x:.1f}, {eye_y:.1f}, {eye_z:.1f})cm, "
                f"correction=[{int(a_v)}, {int(a_h)}]"
            )

        return [int(a_v), int(a_h)], eye_position

    def estimate_gaze_angle_v2(
        self,
        le_center: tuple[float, float],
        re_center: tuple[float, float],
        le_iris_points: Optional[list[tuple[float, float, float]]],
        re_iris_points: Optional[list[tuple[float, float, float]]],
        video_size: tuple[int, int],
    ) -> tuple[list[int], list[float]]:
        """
        Estimate gaze redirection angles using pure iris position tracking.

        This method detects where the person is looking by tracking the iris
        center position relative to the eye center, then applies the opposite
        correction to make them appear to look at the camera.

        No camera geometry is needed - if you're looking 15° up, we correct 15° down.
        A manual offset (adjustable via WASD) handles any systematic bias.

        Args:
            le_center: Left eye center (x, y) in pixels
            re_center: Right eye center (x, y) in pixels
            le_iris_points: Left eye iris points (4 points with x, y, z), or None
            re_iris_points: Right eye iris points (4 points with x, y, z), or None
            video_size: (width, height) of video frame

        Returns:
            (alpha [vertical, horizontal], eye_position [x, y, z])
        """
        # Get eye position estimate (for visualization only, not used in correction)
        settings = self.camera_settings
        ipd_pixels = np.sqrt(
            (le_center[0] - re_center[0]) ** 2 + (le_center[1] - re_center[1]) ** 2
        )
        eye_z = -(settings.focal_length * settings.ipd) / ipd_pixels
        eye_x = (
            -abs(eye_z)
            * (le_center[0] + re_center[0] - video_size[0])
            / (2 * settings.focal_length)
        )
        eye_y = (
            abs(eye_z)
            * (le_center[1] + re_center[1] - video_size[1])
            / (2 * settings.focal_length)
        )
        eye_position = [eye_x, eye_y, eye_z]
        self.last_eye_position = eye_position

        # Validate iris points - fall back to v1 if invalid
        for i in range(4):
            if (np.isnan(le_iris_points[i][0]) or np.isnan(le_iris_points[i][1]) or
                np.isnan(re_iris_points[i][0]) or np.isnan(re_iris_points[i][1])):
                if self.debug_v2:
                    self.logger.log(f"v2: NaN in iris points at index {i}, falling back to v1")
                return self.estimate_gaze_angle(le_center, re_center, video_size)

        # Compute iris centers for both eyes
        le_iris_cx = sum(p[0] for p in le_iris_points) / 4
        le_iris_cy = sum(p[1] for p in le_iris_points) / 4
        re_iris_cx = sum(p[0] for p in re_iris_points) / 4
        re_iris_cy = sum(p[1] for p in re_iris_points) / 4

        # Compute displacement of iris center from eye center (in pixels)
        # Positive dx = iris is to the right of eye center = looking right
        # Positive dy = iris is below eye center = looking down
        le_dx = le_iris_cx - le_center[0]
        le_dy = le_iris_cy - le_center[1]
        re_dx = re_iris_cx - re_center[0]
        re_dy = re_iris_cy - re_center[1]

        # Average the displacements from both eyes
        avg_dx = (le_dx + re_dx) / 2
        avg_dy = (le_dy + re_dy) / 2

        # Estimate eye width for scaling (distance between eye centers ~3x eye width)
        eye_distance = np.sqrt((le_center[0] - re_center[0])**2 + (le_center[1] - re_center[1])**2)
        estimated_eye_width = eye_distance / 3  # Rough approximation

        # Convert pixel displacement to approximate gaze angle
        # When iris moves ~1/4 of eye width, that corresponds to gaze_scale degrees of shift
        gaze_scale = settings.gaze_scale / (estimated_eye_width / 4)

        # Current gaze direction (where person is looking relative to straight ahead)
        detected_gaze_h = avg_dx * gaze_scale  # Horizontal: positive = looking right
        detected_gaze_v = avg_dy * gaze_scale  # Vertical: positive = looking down

        # Store raw gaze for calibration and detected gaze for display
        self.last_raw_gaze = (detected_gaze_v, detected_gaze_h)
        self.last_detected_gaze = (detected_gaze_v, detected_gaze_h)

        # Pure iris-based correction: apply OPPOSITE of detected gaze + manual offset
        # Looking up (negative detected_gaze_v)? Apply positive correction (warp down to center)
        # Looking right (positive detected_gaze_h)? Apply negative correction (warp left to center)
        manual_v = settings.gaze_offset_v
        manual_h = settings.gaze_offset_h
        a_v = -detected_gaze_v + manual_v
        a_h = -detected_gaze_h + manual_h

        if self.debug_v2:
            self.logger.log(
                f"v2: iris_disp=({avg_dx:.1f}, {avg_dy:.1f})px, "
                f"detected_gaze=({detected_gaze_h:.1f}°, {detected_gaze_v:.1f}°), "
                f"manual=({manual_h:.1f}°, {manual_v:.1f}°), "
                f"correction=[{int(a_v)}, {int(a_h)}]"
            )

        # Store for visualization
        self.last_eye_center_pixels = (
            (le_center[0] + re_center[0]) / 2,
            (le_center[1] + re_center[1]) / 2
        )

        # Compute visualization vectors (normalized gaze direction)
        a_v_rad = math.radians(a_v)
        a_h_rad = math.radians(a_h)
        self.last_gaze_direction = np.array([
            math.sin(a_h_rad),
            math.sin(a_v_rad),
            math.cos(a_v_rad) * math.cos(a_h_rad)
        ])
        self.last_eye_to_camera = np.array([0, 0, 1])  # Camera is straight ahead

        # Clamp to reasonable range
        a_v = np.clip(a_v, -45, 45)
        a_h = np.clip(a_h, -45, 45)

        return [int(a_v), int(a_h)], eye_position

    def _compute_iris_plane_normal(
        self,
        iris_points: list[tuple[float, float, float]],
        video_size: tuple[int, int],
    ) -> np.ndarray:
        """
        Compute the normal vector of the iris plane from 4 iris landmarks.

        The iris plane normal points outward from the eye, indicating gaze direction.
        Iris points order: [Top, Right, Bottom, Left]

        Args:
            iris_points: 4 iris landmark points with (x, y, z) coordinates
            video_size: (width, height) for normalizing coordinates

        Returns:
            Unit normal vector (nx, ny, nz) pointing in gaze direction
        """
        # Extract points - order is [Top, Right, Bottom, Left]
        top = np.array(iris_points[0])
        right = np.array(iris_points[1])
        bottom = np.array(iris_points[2])
        left = np.array(iris_points[3])

        # Normalize x, y to centered coordinates (z is already relative depth)
        w, h = video_size
        for pt in [top, right, bottom, left]:
            pt[0] = (pt[0] - w / 2) / w  # Center and normalize x
            pt[1] = (pt[1] - h / 2) / h  # Center and normalize y
            # z from MediaPipe is already a relative depth value

        # Compute vectors on the iris plane
        v_vertical = top - bottom    # Points upward on iris
        v_horizontal = right - left  # Points rightward on iris

        # Cross product gives normal vector
        # Order matters for direction: we want outward-pointing normal
        normal = np.cross(v_horizontal, v_vertical)

        # Normalize to unit vector
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
        else:
            # Fallback: assume looking straight ahead
            normal = np.array([0.0, 0.0, 1.0])

        return normal

    def estimate_gaze_angle_v3(
        self,
        le_center: tuple[float, float],
        re_center: tuple[float, float],
        le_iris_points: list[tuple[float, float, float]],
        re_iris_points: list[tuple[float, float, float]],
        video_size: tuple[int, int],
        head_pose: Optional[object] = None,
    ) -> tuple[list[int], list[float]]:
        """
        Estimate gaze redirection angles using 2D iris displacement with calibration
        and optional head pose compensation.

        This method uses the reliable 2D iris displacement approach (how far the
        iris center is from the eye center) combined with calibration support.
        When head pose is available, it compensates for head rotation to get
        the true gaze direction relative to the camera.

        Gaze = Head pose + Eye-in-head gaze
        - iris_gaze: where eyes are looking relative to head
        - head_pose: where head is pointing relative to camera
        - actual_gaze = head_pose + iris_gaze

        When calibrated (neutral_gaze is set), the stored neutral position is
        subtracted, making corrections relative to the "looking at camera" position.

        Args:
            le_center: Left eye center (x, y) in pixels
            re_center: Right eye center (x, y) in pixels
            le_iris_points: Left eye iris points (4 points with x, y, z)
            re_iris_points: Right eye iris points (4 points with x, y, z)
            video_size: (width, height) of video frame
            head_pose: Optional HeadPose with pitch, yaw, roll in degrees

        Returns:
            (alpha [vertical, horizontal], eye_position [x, y, z])
        """
        settings = self.camera_settings

        # Compute eye position for visualization
        ipd_pixels = np.sqrt(
            (le_center[0] - re_center[0]) ** 2 + (le_center[1] - re_center[1]) ** 2
        )
        eye_z = -(settings.focal_length * settings.ipd) / ipd_pixels
        eye_x = (
            -abs(eye_z)
            * (le_center[0] + re_center[0] - video_size[0])
            / (2 * settings.focal_length)
        )
        eye_y = (
            abs(eye_z)
            * (le_center[1] + re_center[1] - video_size[1])
            / (2 * settings.focal_length)
        )
        eye_position = [eye_x, eye_y, eye_z]
        self.last_eye_position = eye_position

        # Validate iris points
        for i in range(4):
            for pts in [le_iris_points, re_iris_points]:
                if np.isnan(pts[i][0]) or np.isnan(pts[i][1]):
                    if self.debug_v2:
                        self.logger.log(f"v3: NaN in iris points, falling back to v1")
                    return self.estimate_gaze_angle(le_center, re_center, video_size)

        # Compute iris centers for both eyes
        le_iris_cx = sum(p[0] for p in le_iris_points) / 4
        le_iris_cy = sum(p[1] for p in le_iris_points) / 4
        re_iris_cx = sum(p[0] for p in re_iris_points) / 4
        re_iris_cy = sum(p[1] for p in re_iris_points) / 4

        # Compute displacement of iris center from eye center (in pixels)
        # Positive dx = iris is to the right of eye center = looking right
        # Positive dy = iris is below eye center = looking down
        le_dx = le_iris_cx - le_center[0]
        le_dy = le_iris_cy - le_center[1]
        re_dx = re_iris_cx - re_center[0]
        re_dy = re_iris_cy - re_center[1]

        # Average the displacements from both eyes
        avg_dx = (le_dx + re_dx) / 2
        avg_dy = (le_dy + re_dy) / 2

        # Store current iris displacement as a pseudo "normal" for calibration
        # We use (dx, dy, 1) normalized as our gaze direction proxy
        disp_magnitude = math.sqrt(avg_dx**2 + avg_dy**2 + 1)
        self.current_iris_normal = np.array([avg_dx / disp_magnitude, avg_dy / disp_magnitude, 1 / disp_magnitude])

        # Estimate eye width for scaling (distance between eye centers ~3x eye width)
        eye_distance = np.sqrt((le_center[0] - re_center[0])**2 + (le_center[1] - re_center[1])**2)
        estimated_eye_width = eye_distance / 3  # Rough approximation

        # Convert pixel displacement to approximate gaze angle
        # When iris moves ~1/4 of eye width, that corresponds to gaze_scale degrees of shift
        pixels_per_degree = (estimated_eye_width / 4) / settings.gaze_scale

        # Iris gaze in degrees (where eyes are looking relative to HEAD)
        iris_gaze_h = avg_dx / pixels_per_degree  # Horizontal: positive = looking right
        iris_gaze_v = avg_dy / pixels_per_degree  # Vertical: positive = looking down

        # Apply head pose compensation if available
        # Total gaze = head pose + eye-in-head gaze
        # head_pose.yaw: positive = head turned right
        # head_pose.pitch: positive = head tilted down
        try:
            if head_pose is not None:
                head_yaw = getattr(head_pose, 'yaw', 0.0)
                head_pitch = getattr(head_pose, 'pitch', 0.0)
                head_roll = getattr(head_pose, 'roll', 0.0)
                self.last_head_pose = (head_pitch, head_yaw, head_roll)
                raw_gaze_h = iris_gaze_h + head_yaw
                raw_gaze_v = iris_gaze_v + head_pitch
            else:
                self.last_head_pose = None
                raw_gaze_h = iris_gaze_h
                raw_gaze_v = iris_gaze_v
        except Exception:
            # Head pose handling failed, continue without it
            self.last_head_pose = None
            raw_gaze_h = iris_gaze_h
            raw_gaze_v = iris_gaze_v

        # Store raw gaze for calibration capture (includes head pose)
        self.last_raw_gaze = (raw_gaze_v, raw_gaze_h)

        # Apply calibration offset if calibrated
        # neutral_gaze stores (neutral_v, neutral_h) - the gaze angles when looking at camera
        if settings.neutral_gaze is not None:
            neutral_v, neutral_h = settings.neutral_gaze[0], settings.neutral_gaze[1]
            detected_gaze_v = raw_gaze_v - neutral_v
            detected_gaze_h = raw_gaze_h - neutral_h
        else:
            # Not calibrated - raw gaze is the detected gaze
            detected_gaze_v = raw_gaze_v
            detected_gaze_h = raw_gaze_h

        # Store detected gaze for display and auto-calibration
        self.last_detected_gaze = (detected_gaze_v, detected_gaze_h)

        # Apply OPPOSITE correction + manual offset
        # Looking up (negative detected_gaze_v)? Apply positive correction (warp down to center)
        # Looking right (positive detected_gaze_h)? Apply negative correction (warp left to center)
        a_v = -detected_gaze_v + settings.gaze_offset_v
        a_h = -detected_gaze_h + settings.gaze_offset_h

        if self.debug_v2:
            head_info = ""
            if head_pose is not None:
                head_yaw = getattr(head_pose, 'yaw', 0.0)
                head_pitch = getattr(head_pose, 'pitch', 0.0)
                head_info = f"head=({head_pitch:.1f}°, {head_yaw:.1f}°), "
            self.logger.log(
                f"v3: iris_disp=({avg_dx:.1f}, {avg_dy:.1f})px, "
                f"iris_gaze=({iris_gaze_h:.1f}°, {iris_gaze_v:.1f}°), "
                f"{head_info}"
                f"raw_gaze=({raw_gaze_h:.1f}°, {raw_gaze_v:.1f}°), "
                f"detected=({detected_gaze_h:.1f}°, {detected_gaze_v:.1f}°), "
                f"correction=[{int(a_v)}, {int(a_h)}]"
            )

        # Store for visualization
        self.last_eye_center_pixels = (
            (le_center[0] + re_center[0]) / 2,
            (le_center[1] + re_center[1]) / 2
        )

        # Compute visualization vectors
        a_v_rad = math.radians(detected_gaze_v)
        a_h_rad = math.radians(detected_gaze_h)
        self.last_gaze_direction = np.array([
            math.sin(a_h_rad),
            math.sin(a_v_rad),
            math.cos(a_v_rad) * math.cos(a_h_rad)
        ])
        self.last_eye_to_camera = np.array([0, 0, 1])  # Camera is straight ahead

        # Clamp to reasonable range
        a_v = np.clip(a_v, -45, 45)
        a_h = np.clip(a_h, -45, 45)

        return [int(a_v), int(a_h)], eye_position

    def correct_eye(
        self, eye_data, eye_side: str, angle: list[int]
    ) -> np.ndarray:
        """
        Apply gaze correction to a single eye.

        Args:
            eye_data: Eye extraction data (EyeData from face_predictor)
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

    def apply_correction(self, frame: np.ndarray, face_data, video_size: tuple[int, int]) -> np.ndarray:
        """
        Apply gaze correction to a frame using extracted face data.

        Args:
            frame: BGR video frame to modify
            face_data: Extracted face/eye data from FacePredictor (FaceData)
            video_size: (width, height) of video frame

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

        # Extract head pose if available
        head_pose = getattr(face_data, 'head_pose', None)

        # Estimate gaze angle using v3 (2D iris displacement with calibration) if iris_points available
        # Falls back to v1 (geometric) if iris points have NaN or are unavailable
        if le.iris_points is not None and re.iris_points is not None:
            alpha, _ = self.estimate_gaze_angle_v3(
                le.center, re.center, le.iris_points, re.iris_points, video_size,
                head_pose=head_pose
            )
        else:
            alpha, _ = self.estimate_gaze_angle(le.center, re.center, video_size)

        # Correct both eyes
        le_corrected = self.correct_eye(le, "L", alpha)
        re_corrected = self.correct_eye(re, "R", alpha)

        # Replace eye regions with alpha blending at edges
        self._blend_eye_region(frame, le_corrected, le.top_left, le.original_size)
        self._blend_eye_region(frame, re_corrected, re.top_left, re.original_size)

        return frame

    def _blend_eye_region(
        self,
        frame: np.ndarray,
        corrected: np.ndarray,
        top_left: tuple[int, int],
        original_size: tuple[int, int],
    ) -> None:
        """
        Blend corrected eye region into frame with smooth edges.

        Args:
            frame: Target frame to modify in-place
            corrected: Corrected eye image in [0, 1] range
            top_left: (row, col) position in frame
            original_size: (height, width) of original eye region
        """
        pc = self.pixel_cut
        bb = self.blend_border

        # Get the region coordinates
        r1, r2 = top_left[0] + pc[0], top_left[0] + original_size[0] - pc[0]
        c1, c2 = top_left[1] + pc[1], top_left[1] + original_size[1] - pc[1]

        # Get cropped corrected eye
        eye_cropped = (corrected[pc[0]:-pc[0], pc[1]:-pc[1]] * 255).astype(np.uint8)

        # Create alpha mask with feathered edges using distance from border
        h, w = eye_cropped.shape[:2]

        # Create distance-based alpha mask
        y_dist = np.minimum(np.arange(h), np.arange(h-1, -1, -1)).reshape(-1, 1)
        x_dist = np.minimum(np.arange(w), np.arange(w-1, -1, -1)).reshape(1, -1)
        dist = np.minimum(y_dist, x_dist).astype(np.float32)

        # Normalize to [0, 1] based on blend border
        alpha = np.clip(dist / bb, 0, 1)

        # Expand alpha to 3 channels
        alpha_3ch = np.expand_dims(alpha, axis=2)

        # Blend
        original_region = frame[r1:r2, c1:c2].astype(np.float32)
        blended = alpha_3ch * eye_cropped + (1 - alpha_3ch) * original_region
        frame[r1:r2, c1:c2] = blended.astype(np.uint8)

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
