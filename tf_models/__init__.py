"""
TensorFlow Models Package for Gaze Correction

This package provides versioned gaze correction models.

Versions:
    - gaze_corrector_v1: Initial DeepWarp-based gaze correction model
"""

# Import versioned models
from tf_models import gaze_corrector_v1

# Backward compatibility: expose v1 as default
from tf_models.gaze_corrector_v1 import gaze_warp_model
from tf_models.gaze_corrector_v1 import layers
from tf_models.gaze_corrector_v1 import spatial_transform
from tf_models.gaze_corrector_v1 import ModelConfig
from tf_models.gaze_corrector_v1 import build_inference_graph

# Additional backward compatibility aliases
from tf_models.gaze_corrector_v1 import gaze_warp_model as flx
from tf_models.gaze_corrector_v1 import layers as tf_utils
from tf_models.gaze_corrector_v1 import spatial_transform as transformation

__all__ = [
    "gaze_corrector_v1",
    "gaze_warp_model",
    "layers", 
    "spatial_transform",
    "ModelConfig",
    "build_inference_graph",
    # Backward compatibility
    "flx",
    "tf_utils",
    "transformation",
]
