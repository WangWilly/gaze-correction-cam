"""
TensorFlow Models Package for Gaze Correction

This package provides neural network components for real-time gaze redirection
in video, based on the DeepWarp approach.

Modules:
    - gaze_warp_model: Main gaze correction model architecture
    - layers: Reusable neural network layer building blocks
    - spatial_transform: Differentiable image warping operations

Quick Start:
    >>> from tf_models import gaze_warp_model
    >>> 
    >>> config = gaze_warp_model.ModelConfig()
    >>> with tf.Graph().as_default():
    ...     pred, flow, lcm = gaze_warp_model.build_inference_graph(
    ...         eye_image, anchor_points, gaze_angles, is_training=False, config=config
    ...     )

For backward compatibility, the old module names are also available:
    - flx (alias for gaze_warp_model)
    - tf_utils (alias for layers)
    - transformation (alias for spatial_transform)
"""

# New module names (preferred)
from tf_models import gaze_warp_model
from tf_models import layers
from tf_models import spatial_transform

# Model configuration
from tf_models.gaze_warp_model import ModelConfig

# Key functions for inference
from tf_models.gaze_warp_model import build_inference_graph

# Backward compatibility aliases
from tf_models import gaze_warp_model as flx
from tf_models import layers as tf_utils
from tf_models import spatial_transform as transformation

__all__ = [
    # New modules
    "gaze_warp_model",
    "layers", 
    "spatial_transform",
    # Key exports
    "ModelConfig",
    "build_inference_graph",
    # Backward compatibility
    "flx",
    "tf_utils",
    "transformation",
]
