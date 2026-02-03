"""
Face Predictor Interface Module

This module defines the abstract interface for face predictors and provides
implementations for dlib and mediapipe backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2


################################################################################
# Data Classes for Face Landmarks
################################################################################


@dataclass
class EyeLandmarks:
    """Represents eye landmarks with 6 key points."""

    points: list[tuple[int, int]]  # 6 points for each eye
    center: tuple[float, float]  # Eye center (x, y)
    iris_points: Optional[list[tuple[float, float, float]]] = None  # 4 iris points with (x, y, z), temporary storage before moving to EyeData


@dataclass
class EyeData:
    """
    Contains all data needed for gaze correction model inference.
    
    This is the output of eye extraction and input to gaze correction.
    """

    image: np.ndarray              # Eye image normalized to [0, 1], shape (H, W, 3)
    anchor_map: np.ndarray         # Feature point map, shape (H, W, ef_dim)
    original_size: tuple[int, int] # (height, width) before resize
    top_left: tuple[int, int]      # (row, col) position in original frame
    center: tuple[float, float]    # Eye center (x, y) in original frame
    iris_points: Optional[list[tuple[float, float, float]]] = None  # 4 iris points (x, y, z) for gaze direction computation


@dataclass
class FaceData:
    """Container for extracted face data including both eyes."""

    left_eye: Optional[EyeData]
    right_eye: Optional[EyeData]
    landmarks: Optional["FaceLandmarks"] = None


@dataclass
class FaceLandmarks:
    """Container for detected face landmarks."""

    left_eye: EyeLandmarks  # Left eye (from viewer's perspective, landmarks 42-47)
    right_eye: EyeLandmarks  # Right eye (from viewer's perspective, landmarks 36-41)
    raw_shape: Optional[object] = None  # Original shape object for compatibility


################################################################################
# Abstract Face Predictor Interface
################################################################################


@dataclass
class EyeExtractionConfig:
    """Configuration for eye region extraction."""

    input_size: tuple[int, int] = (48, 64)  # (height, width) for model input
    ef_dim: int = 12  # Feature dimension (6 points * 2 coords)


class FacePredictor(ABC):
    """Abstract base class for face prediction backends."""

    @abstractmethod
    def list_eye_data(
        self,
        frame: np.ndarray,
        config: EyeExtractionConfig,
    ) -> list[FaceData]:
        """
        Detect faces and extract eye data for gaze correction.

        This is the main public interface that combines face detection,
        landmark prediction, and eye extraction into a single call.

        Args:
            frame: BGR video frame
            config: Eye extraction configuration

        Returns:
            List of FaceData objects for each detected face
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this predictor backend."""
        pass


################################################################################
# Dlib Face Predictor Implementation
################################################################################


class DlibFacePredictor(FacePredictor):
    """Face predictor using dlib's 68-point landmark model."""

    # Landmark indices for dlib 68-point model
    RIGHT_EYE_INDICES = [36, 37, 38, 39, 40, 41]  # Right eye from viewer's perspective
    LEFT_EYE_INDICES = [42, 43, 44, 45, 46, 47]  # Left eye from viewer's perspective

    def __init__(self, predictor_path: str = "./lm_feat/shape_predictor_68_face_landmarks.dat"):
        """
        Initialize dlib face predictor.

        Args:
            predictor_path: Path to dlib shape predictor model file
        """
        import dlib

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self._dlib = dlib  # Keep reference for rectangle creation

    def _detect_faces(self, gray_frame: np.ndarray) -> list:
        """Detect faces using dlib's HOG-based detector."""
        return self.detector(gray_frame, 0)

    def _predict_landmarks(
        self, gray_frame: np.ndarray, face_bbox, frame_scale: tuple[float, float]
    ) -> Optional[FaceLandmarks]:
        """Predict 68 facial landmarks and extract eye regions."""
        x_ratio, y_ratio = frame_scale

        # Scale bounding box to full frame size
        scaled_bbox = self._dlib.rectangle(
            left=int(face_bbox.left() * x_ratio),
            right=int(face_bbox.right() * x_ratio),
            top=int(face_bbox.top() * y_ratio),
            bottom=int(face_bbox.bottom() * y_ratio),
        )

        shape = self.predictor(gray_frame, scaled_bbox)

        # Extract left eye landmarks (42-47)
        left_eye_points = [
            (shape.part(i).x, shape.part(i).y) for i in self.LEFT_EYE_INDICES
        ]
        left_center = self._compute_eye_center(shape, 42, 45)
        left_eye = EyeLandmarks(points=left_eye_points, center=left_center)

        # Extract right eye landmarks (36-41)
        right_eye_points = [
            (shape.part(i).x, shape.part(i).y) for i in self.RIGHT_EYE_INDICES
        ]
        right_center = self._compute_eye_center(shape, 36, 39)
        right_eye = EyeLandmarks(points=right_eye_points, center=right_center)

        return FaceLandmarks(
            left_eye=left_eye, right_eye=right_eye, raw_shape=shape
        )

    def _compute_eye_center(
        self, shape, left_corner_idx: int, right_corner_idx: int
    ) -> tuple[float, float]:
        """Compute eye center from corner landmarks."""
        cx = (shape.part(left_corner_idx).x + shape.part(right_corner_idx).x) * 0.5
        cy = (shape.part(left_corner_idx).y + shape.part(right_corner_idx).y) * 0.5
        return (cx, cy)

    def _extract_eye_data(
        self,
        frame: np.ndarray,
        landmarks: FaceLandmarks,
        config: EyeExtractionConfig,
    ) -> FaceData:
        """Extract eye regions for gaze correction."""
        left_eye = self._extract_single_eye(
            frame, landmarks.left_eye, "L", config
        )
        right_eye = self._extract_single_eye(
            frame, landmarks.right_eye, "R", config
        )
        return FaceData(left_eye=left_eye, right_eye=right_eye, landmarks=landmarks)

    def _extract_single_eye(
        self,
        frame: np.ndarray,
        eye_landmarks: EyeLandmarks,
        eye_side: str,
        config: EyeExtractionConfig,
    ) -> Optional[EyeData]:
        """
        Extract a single eye region and create anchor map.
        
        Args:
            frame: BGR video frame
            eye_landmarks: Eye landmark points
            eye_side: "L" or "R"
            config: Extraction configuration
        """
        size_I = config.input_size
        points = eye_landmarks.points
        eye_cx, eye_cy = eye_landmarks.center

        # Determine landmark order for anchor map
        # Left eye (dlib 42-47): use order [3,2,1,0,5,4] -> [45,44,43,42,47,46]
        # Right eye (dlib 36-41): use order [0,1,2,3,4,5] -> [36,37,38,39,40,41]
        if eye_side == "L":
            fp_seq = [3, 2, 1, 0, 5, 4]
        else:
            fp_seq = [0, 1, 2, 3, 4, 5]

        # Calculate bounding box
        eye_len = abs(points[3][0] - points[0][0])
        bx_half_w = eye_len * 3 / 4
        bx_h = 1.5 * bx_half_w
        sft_up = bx_h * 7 / 12
        sft_low = bx_h * 5 / 12

        top = int(eye_cy - sft_up)
        bottom = int(eye_cy + sft_low)
        left = int(eye_cx - bx_half_w)
        right = int(eye_cx + bx_half_w)

        # Extract and validate eye region
        img_eye = frame[top:bottom, left:right]
        if img_eye.size == 0:
            return None

        ori_size = (img_eye.shape[0], img_eye.shape[1])
        lt_coord = (top, left)

        img_eye = cv2.resize(img_eye, (size_I[1], size_I[0]))

        # Create anchor maps
        ach_map = None
        for i, idx in enumerate(fp_seq):
            pt = points[idx]
            resize_x = int((pt[0] - lt_coord[1]) * size_I[1] / ori_size[1])
            resize_y = int((pt[1] - lt_coord[0]) * size_I[0] / ori_size[0])

            ach_map_y = np.expand_dims(
                np.expand_dims(np.arange(0, size_I[0]) - resize_y, axis=1), axis=2
            )
            ach_map_y = np.tile(ach_map_y, [1, size_I[1], 1])

            ach_map_x = np.expand_dims(
                np.expand_dims(np.arange(0, size_I[1]) - resize_x, axis=0), axis=2
            )
            ach_map_x = np.tile(ach_map_x, [size_I[0], 1, 1])

            if ach_map is None:
                ach_map = np.concatenate((ach_map_x, ach_map_y), axis=2)
            else:
                ach_map = np.concatenate((ach_map, ach_map_x, ach_map_y), axis=2)

        return EyeData(
            image=img_eye / 255.0,
            anchor_map=ach_map,
            original_size=ori_size,
            top_left=lt_coord,
            center=eye_landmarks.center,
        )

    def list_eye_data(
        self,
        frame: np.ndarray,
        config: EyeExtractionConfig,
    ) -> list[FaceData]:
        """Detect faces and extract eye data for gaze correction."""
        results: list[FaceData] = []
        
        # Detect faces on grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(gray)
        
        for face in faces:
            # Get landmarks (no scaling needed when using full frame)
            landmarks = self._predict_landmarks(gray, face, (1.0, 1.0))
            if landmarks is None:
                continue
            
            # Extract eye data
            face_data = self._extract_eye_data(frame, landmarks, config)
            results.append(face_data)
        
        return results

    def get_name(self) -> str:
        return "dlib"


################################################################################
# MediaPipe Face Predictor Implementation (Placeholder)
################################################################################


class MediaPipeFacePredictor(FacePredictor):
    """Face predictor using MediaPipe's face mesh model."""

    # MediaPipe face mesh indices for eyes (approximate mapping to dlib-style)
    # Left eye (from viewer's perspective): 362, 385, 387, 263, 373, 380
    # Right eye (from viewer's perspective): 33, 160, 158, 133, 153, 144
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

    # Eye center calculation using iris landmarks
    # Left iris: 473 (center), 474, 475, 476, 477
    # Right iris: 468 (center), 469, 470, 471, 472
    LEFT_EYE_CORNERS = (474, 476)
    RIGHT_EYE_CORNERS = (471, 469)

    # 4 iris landmarks for computing eye plane normal vector (excluding center)
    # These form a rough plane around the iris
    LEFT_IRIS_PLANE_INDICES = [475, 476, 477, 474]  # 4 points around right iris [Top, Right, Bottom, Left]
    RIGHT_IRIS_PLANE_INDICES = [470, 471, 472, 469] # 4 points around left iris  [Top, Right, Bottom, Left]

    def __init__(self, model_path: str = "./models/face_landmarker.task"):
        """
        Initialize MediaPipe face predictor.

        Args:
            model_path: Path to MediaPipe face landmarker model
        """
        import mediapipe as mp
        from time import time

        self._mp = mp
        self._start_time = time()

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def _detect_landmarks(self, bgr_frame: np.ndarray) -> list[FaceLandmarks]:
        """
        Detect faces and extract landmarks from a BGR frame.

        Args:
            bgr_frame: BGR video frame

        Returns:
            List of FaceLandmarks for each detected face
        """
        from time import time

        h, w = bgr_frame.shape[:2]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=bgr_frame)
        timestamp_ms = int((time() - self._start_time) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return []

        face_landmarks_list: list[FaceLandmarks] = []
        for landmarks in result.face_landmarks:
            face_lm = self._extract_eye_landmarks(landmarks, w, h)
            face_landmarks_list.append(face_lm)

        return face_landmarks_list

    def _extract_eye_landmarks(
        self, landmarks, frame_width: int, frame_height: int
    ) -> FaceLandmarks:
        """
        Extract eye landmarks from MediaPipe face landmarks.

        Args:
            landmarks: MediaPipe normalized landmarks for a single face
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            FaceLandmarks with left and right eye data
        """
        w, h = frame_width, frame_height

        # Extract left eye landmarks
        left_eye_points = [
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in self.LEFT_EYE_INDICES
        ]
        left_center = (
            (landmarks[self.LEFT_EYE_CORNERS[0]].x + landmarks[self.LEFT_EYE_CORNERS[1]].x) * w / 2,
            (landmarks[self.LEFT_EYE_CORNERS[0]].y + landmarks[self.LEFT_EYE_CORNERS[1]].y) * h / 2,
        )
        # Extract left iris points for gaze direction (with z coordinate)
        left_iris_points = [
            (landmarks[i].x * w, landmarks[i].y * h, landmarks[i].z)
            for i in self.LEFT_IRIS_PLANE_INDICES
        ]
        left_eye = EyeLandmarks(
            points=left_eye_points, center=left_center, iris_points=left_iris_points
        )

        # Extract right eye landmarks
        right_eye_points = [
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in self.RIGHT_EYE_INDICES
        ]
        right_center = (
            (landmarks[self.RIGHT_EYE_CORNERS[0]].x + landmarks[self.RIGHT_EYE_CORNERS[1]].x) * w / 2,
            (landmarks[self.RIGHT_EYE_CORNERS[0]].y + landmarks[self.RIGHT_EYE_CORNERS[1]].y) * h / 2,
        )
        # Extract right iris points for gaze direction (with z coordinate)
        right_iris_points = [
            (landmarks[i].x * w, landmarks[i].y * h, landmarks[i].z)
            for i in self.RIGHT_IRIS_PLANE_INDICES
        ]
        right_eye = EyeLandmarks(
            points=right_eye_points, center=right_center, iris_points=right_iris_points
        )

        return FaceLandmarks(left_eye=left_eye, right_eye=right_eye, raw_shape=landmarks)

    def _extract_eye_data(
        self,
        frame: np.ndarray,
        landmarks: FaceLandmarks,
        config: EyeExtractionConfig,
    ) -> FaceData:
        """Extract eye regions for gaze correction."""
        left_eye = self._extract_single_eye(
            frame, landmarks.left_eye, "L", config
        )
        right_eye = self._extract_single_eye(
            frame, landmarks.right_eye, "R", config
        )
        return FaceData(left_eye=left_eye, right_eye=right_eye, landmarks=landmarks)

    def _extract_single_eye(
        self,
        frame: np.ndarray,
        eye_landmarks: EyeLandmarks,
        eye_side: str,
        config: EyeExtractionConfig,
    ) -> Optional[EyeData]:
        """Extract a single eye region and create anchor map."""
        size_I = config.input_size
        points = eye_landmarks.points
        eye_cx = sum([pt[0] for pt in points]) / len(points)
        eye_cy = sum([pt[1] for pt in points]) / len(points)

        # MediaPipe uses same logic as dlib for anchor map ordering
        if eye_side == "L":
            fp_seq = [3, 2, 1, 0, 5, 4]
        else:
            fp_seq = [0, 1, 2, 3, 4, 5]

        # Calculate bounding box
        eye_len = abs(points[3][0] - points[0][0])
        bx_half_w = eye_len * 3 / 4
        bx_h = 1.5 * bx_half_w
        sft_up = bx_h * 7 / 12
        sft_low = bx_h * 5 / 12

        top = int(eye_cy - sft_up)
        bottom = int(eye_cy + sft_low)
        left = int(eye_cx - bx_half_w)
        right = int(eye_cx + bx_half_w)

        img_eye = frame[top:bottom, left:right]
        if img_eye.size == 0:
            return None

        ori_size = (img_eye.shape[0], img_eye.shape[1])
        lt_coord = (top, left)

        img_eye = cv2.resize(img_eye, (size_I[1], size_I[0]))

        # Create anchor maps
        ach_map = None
        for _, idx in enumerate(fp_seq):
            pt = points[idx]
            resize_x = int((pt[0] - lt_coord[1]) * size_I[1] / ori_size[1])
            resize_y = int((pt[1] - lt_coord[0]) * size_I[0] / ori_size[0])

            ach_map_y = np.expand_dims(
                np.expand_dims(np.arange(0, size_I[0]) - resize_y, axis=1), axis=2
            )
            ach_map_y = np.tile(ach_map_y, [1, size_I[1], 1])

            ach_map_x = np.expand_dims(
                np.expand_dims(np.arange(0, size_I[1]) - resize_x, axis=0), axis=2
            )
            ach_map_x = np.tile(ach_map_x, [size_I[0], 1, 1])

            if ach_map is None:
                ach_map = np.concatenate((ach_map_x, ach_map_y), axis=2)
            else:
                ach_map = np.concatenate((ach_map, ach_map_x, ach_map_y), axis=2)

        return EyeData(
            image=img_eye / 255.0,
            anchor_map=ach_map,
            original_size=ori_size,
            top_left=lt_coord,
            center=eye_landmarks.center,
            iris_points=eye_landmarks.iris_points,
        )

    def list_eye_data(
        self,
        frame: np.ndarray,
        config: EyeExtractionConfig,
    ) -> list[FaceData]:
        """
        Detect faces and extract eye data for gaze correction.

        Args:
            frame: BGR video frame
            config: Eye extraction configuration

        Returns:
            List of FaceData objects for each detected face
        """
        # Detect all face landmarks in a single call
        all_face_landmarks = self._detect_landmarks(frame)

        # Extract eye data for each detected face
        results: list[FaceData] = []
        for landmarks in all_face_landmarks:
            face_data = self._extract_eye_data(frame, landmarks, config)
            results.append(face_data)

        return results

    def get_name(self) -> str:
        return "mediapipe"


################################################################################
# Factory Function
################################################################################


def create_face_predictor(
    backend: str = "dlib",
    predictor_path: Optional[str] = None,
) -> FacePredictor:
    """
    Factory function to create a face predictor.

    Args:
        backend: "dlib" or "mediapipe"
        predictor_path: Optional path to model file

    Returns:
        FacePredictor instance
    """
    if backend == "dlib":
        path = predictor_path or "./lm_feat/shape_predictor_68_face_landmarks.dat"
        return DlibFacePredictor(path)
    elif backend == "mediapipe":
        path = predictor_path or "./models/face_landmarker.task"
        return MediaPipeFacePredictor(path)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'dlib' or 'mediapipe'.")
