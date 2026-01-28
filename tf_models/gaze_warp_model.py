"""
Gaze Warping Model (DeepWarp-based)

This module implements the neural network architecture for gaze redirection.
The model learns to warp eye images to redirect gaze direction.

Architecture Overview:
    1. Angle Encoder: Encodes gaze angles into spatial feature maps
    2. Warping Module:
        - Coarse Level: Low-resolution flow estimation
        - Fine Level: High-resolution flow refinement
    3. LCM (Light Correction Module): Handles lighting/color adjustments

Based on the DeepWarp approach for eye gaze manipulation.

Usage:
    >>> config = ModelConfig()
    >>> pred_image, flow, lcm = build_inference_graph(
    ...     eye_image, anchor_points, gaze_angle, is_training=False, config=config
    ... )
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import tensorflow as tf

from tf_models import layers
from tf_models import spatial_transform


################################################################################
# Constants
################################################################################

# Pixels to crop from border when computing losses (reduces edge artifacts)
BORDER_CROP_PIXELS = 3


################################################################################
# Configuration
################################################################################


@dataclass
class ModelConfig:
    """
    Configuration for the gaze warping model.
    
    Attributes:
        height: Input image height in pixels
        width: Input image width in pixels
        encoded_angle_dim: Dimension of encoded gaze angle features
    """
    height: int = 48
    width: int = 64
    encoded_angle_dim: int = 16
    
    @classmethod
    def parse_from(cls, general_cfg: argparse.Namespace) -> "ModelConfig":
        """Create config from command-line arguments."""
        config = cls()
        config.height = general_cfg.height
        config.width = general_cfg.width
        config.encoded_angle_dim = general_cfg.encoded_agl_dim
        return config


@dataclass
class LayerConfig:
    """
    Configuration for a neural network module.
    
    Attributes:
        depths: Number of filters for each layer
        kernel_sizes: Kernel sizes for each layer
    """
    depths: Tuple[int, ...]
    kernel_sizes: Tuple[List[int], ...]


################################################################################
# Default Layer Configurations
################################################################################

def get_coarse_layer_config() -> LayerConfig:
    """Configuration for coarse-level flow estimation."""
    return LayerConfig(
        depths=(32, 64, 64, 32, 16),
        kernel_sizes=([5, 5], [3, 3], [3, 3], [3, 3], [1, 1]),
    )


def get_fine_layer_config() -> LayerConfig:
    """Configuration for fine-level flow refinement."""
    return LayerConfig(
        depths=(32, 64, 32, 16, 4),
        kernel_sizes=([5, 5], [3, 3], [3, 3], [3, 3], [1, 1]),
    )


def get_lcm_layer_config() -> LayerConfig:
    """Configuration for light correction module."""
    return LayerConfig(
        depths=(8, 8, 2),
        kernel_sizes=([3, 3], [3, 3], [1, 1]),
    )


################################################################################
# Angle Encoding
################################################################################


def tile_to_spatial_map(
    features: tf.Tensor,
    height: int,
    width: int,
    feature_dim: int,
) -> tf.Tensor:
    """
    Tile a feature vector to create a spatial feature map.
    
    Takes a [batch, features] tensor and creates a [batch, H, W, features]
    tensor by repeating the features at every spatial location.
    
    Args:
        features: Input features [batch, feature_dim]
        height: Output height
        width: Output width
        feature_dim: Number of features (for shape specification)
    
    Returns:
        Spatial feature map [batch, height, width, feature_dim]
    """
    with tf.name_scope("tile_to_spatial_map"):
        batch_size = tf.shape(features)[0]
        num_spatial_positions = height * width
        
        # Tile features across spatial dimensions
        tiled = tf.tile(features, [1, num_spatial_positions])
        
        # Reshape to spatial format
        spatial_map = tf.reshape(tiled, [batch_size, height, width, feature_dim])
        
        return spatial_map


def build_angle_encoder(
    input_angles: tf.Tensor,
    height: int,
    width: int,
    output_dim: int,
) -> tf.Tensor:
    """
    Encode gaze angles into a spatial feature map.
    
    The encoder uses a small MLP to transform the 2D angle (vertical, horizontal)
    into a richer feature representation, then tiles it across spatial dimensions.
    
    Args:
        input_angles: Gaze angles [batch, 2] (vertical, horizontal in degrees)
        height: Output spatial height
        width: Output spatial width
        output_dim: Number of output feature channels
    
    Returns:
        Encoded angle map [batch, height, width, output_dim]
    """
    # Use "encoder" scope for checkpoint compatibility
    with tf.compat.v1.variable_scope("encoder"):
        # MLP to encode angles into richer features
        hidden_1 = layers.dense_block(input_angles, units=16, name="dnn_blk_0")
        hidden_2 = layers.dense_block(hidden_1, units=16, name="dnn_blk_1")
        angle_features = layers.dense_block(hidden_2, units=output_dim, name="dnn_blk_2")
        
        # Tile to create spatial map
        angle_map = tile_to_spatial_map(angle_features, height, width, output_dim)
        
        return angle_map


################################################################################
# Transformation Module (Dense Convolution Block)
################################################################################


def build_transform_module(
    inputs: tf.Tensor,
    config: LayerConfig,
    is_training: bool,
    name: str = "transform_module",
) -> tf.Tensor:
    """
    Build a densely-connected convolutional module for flow estimation.
    
    Uses dense connections (concatenating outputs of previous layers)
    to improve gradient flow and feature reuse.
    
    Architecture:
        conv_1 -> [conv_1, conv_2] -> [conv_1, conv_2, conv_3] -> conv_4 -> conv_5
    
    Args:
        inputs: Input tensor [batch, height, width, channels]
        config: LayerConfig with depths and kernel sizes
        is_training: Whether model is in training mode
        name: Name scope for the module
    
    Returns:
        Output features [batch, height, width, config.depths[-1]]
    """
    with tf.compat.v1.variable_scope(name):
        # First conv block (checkpoint name: cnn_blk_0)
        block_0 = layers.conv_block(
            inputs,
            filters=config.depths[0],
            kernel_size=config.kernel_sizes[0],
            is_training=is_training,
            name="cnn_blk_0",
        )
        
        # Second conv block (checkpoint name: cnn_blk_1)
        block_1 = layers.conv_block(
            block_0,
            filters=config.depths[1],
            kernel_size=config.kernel_sizes[1],
            is_training=is_training,
            name="cnn_blk_1",
        )
        
        # Third conv block with dense connection (checkpoint name: cnn_blk_2)
        concat_1 = tf.concat([block_0, block_1], axis=3)
        block_2 = layers.conv_block(
            concat_1,
            filters=config.depths[2],
            kernel_size=config.kernel_sizes[2],
            is_training=is_training,
            name="cnn_blk_2",
        )
        
        # Fourth conv block with more dense connections (checkpoint name: cnn_blk_3)
        concat_2 = tf.concat([block_0, block_1, block_2], axis=3)
        block_3 = layers.conv_block(
            concat_2,
            filters=config.depths[3],
            kernel_size=config.kernel_sizes[3],
            is_training=is_training,
            name="cnn_blk_3",
        )
        
        # Final 1x1 conv to reduce channels (checkpoint name: cnn_4)
        output = layers.conv2d_layer(
            block_3,
            filters=config.depths[4],
            kernel_size=config.kernel_sizes[4],
            activation=None,
            use_bias=False,
            name="cnn_4",
        )
        
        return output


################################################################################
# Light Correction Module (LCM)
################################################################################


def apply_light_correction(
    warped_image: tf.Tensor,
    light_weights: tf.Tensor,
) -> tf.Tensor:
    """
    Apply light correction to a warped image.
    
    Blends the warped image with a white palette based on learned weights.
    This helps correct for lighting changes during eye warping.
    
    Formula: output = image * weight_image + white * weight_palette
    
    Args:
        warped_image: Warped eye image [batch, H, W, 3]
        light_weights: Blending weights [batch, H, W, 2]
                       Channel 0: weight for original image
                       Channel 1: weight for white palette
    
    Returns:
        Light-corrected image [batch, H, W, 3]
    """
    with tf.name_scope("apply_light_correction"):
        # Split weights for image and palette
        image_weight, palette_weight = tf.split(light_weights, [1, 1], axis=3)
        
        # Expand weights to match RGB channels
        image_weight = tf.tile(image_weight, [1, 1, 1, 3])
        palette_weight = tf.tile(palette_weight, [1, 1, 1, 3])
        
        # White palette (for blending)
        white_palette = tf.ones(tf.shape(warped_image), dtype=tf.float32)
        
        # Blend image with white
        corrected = tf.add(
            tf.multiply(warped_image, image_weight),
            tf.multiply(white_palette, palette_weight)
        )
        
        return corrected


def build_lcm_module(
    inputs: tf.Tensor,
    config: LayerConfig,
    is_training: bool,
    name: str = "lcm_module",
) -> tf.Tensor:
    """
    Build the Light Correction Module.
    
    Predicts per-pixel weights for blending the warped image with
    a white palette, helping to correct lighting artifacts.
    
    Args:
        inputs: Input features [batch, H, W, C]
        config: LayerConfig for the module
        is_training: Whether model is in training mode
        name: Name scope
    
    Returns:
        Soft blending weights [batch, H, W, 2]
    """
    with tf.compat.v1.variable_scope(name):
        # Checkpoint names: cnn_blk_0, cnn_blk_1, cnn_2
        block_0 = layers.conv_block(
            inputs,
            filters=config.depths[0],
            kernel_size=config.kernel_sizes[0],
            is_training=is_training,
            name="cnn_blk_0",
        )
        
        block_1 = layers.conv_block(
            block_0,
            filters=config.depths[1],
            kernel_size=config.kernel_sizes[1],
            is_training=is_training,
            name="cnn_blk_1",
        )
        
        # Final conv to predict 2 weight channels (checkpoint name: cnn_2)
        logits = layers.conv2d_layer(
            block_1,
            filters=config.depths[2],
            kernel_size=config.kernel_sizes[2],
            activation=None,
            use_bias=False,
            name="cnn_2",
        )
        
        # Softmax to get normalized weights
        weights = tf.nn.softmax(logits)
        
        return weights


################################################################################
# Coarse-to-Fine Warping Module
################################################################################


def build_warping_module(
    inputs: tf.Tensor,
    model_config: ModelConfig,
    is_training: bool,
    coarse_config: Optional[LayerConfig] = None,
    fine_config: Optional[LayerConfig] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Build the coarse-to-fine warping module.
    
    Two-stage flow estimation:
    1. Coarse: Estimate flow at half resolution, then upsample
    2. Fine: Refine flow at full resolution
    
    Args:
        inputs: Concatenated [image, anchor_points, angle_map]
        model_config: Model configuration
        is_training: Training mode flag
        coarse_config: Optional custom config for coarse level
        fine_config: Optional custom config for fine level
    
    Returns:
        Tuple of (raw_flow [batch, H, W, 2], lcm_input [batch, H, W, 2])
    """
    coarse_cfg = coarse_config or get_coarse_layer_config()
    fine_cfg = fine_config or get_fine_layer_config()
    
    with tf.compat.v1.variable_scope("warping_module"):
        # === Coarse Level ===
        # Downsample input for faster processing
        downsampled_inputs = layers.average_pooling_2d(
            inputs, pool_size=(2, 2), strides=(2, 2), padding="same"
        )
        
        coarse_output = build_transform_module(
            downsampled_inputs, coarse_cfg, is_training, name="coarse_level"
        )
        coarse_activated = tf.nn.tanh(coarse_output)
        
        # Upsample coarse flow to full resolution
        coarse_upsampled = tf.image.resize(
            coarse_activated,
            (model_config.height, model_config.width),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        
        # Smooth upsampled flow
        coarse_smoothed = layers.average_pooling_2d(
            coarse_upsampled, pool_size=(2, 2), strides=(1, 1), padding="same"
        )
        
        # === Fine Level ===
        # Concatenate original input with coarse flow
        fine_input = tf.concat([inputs, coarse_smoothed], axis=3, name="fine_input")
        
        fine_output = build_transform_module(
            fine_input, fine_cfg, is_training, name="fine_level"
        )
        
        # Split into flow (2 channels) and LCM input (2 channels)
        raw_flow, lcm_input = tf.split(fine_output, [2, 2], axis=3)
        
        return raw_flow, lcm_input


################################################################################
# Main Inference Graph
################################################################################


def build_inference_graph(
    input_image: tf.Tensor,
    anchor_points: tf.Tensor,
    input_angles: tf.Tensor,
    is_training: bool,
    config: ModelConfig,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Build the complete gaze warping inference graph.
    
    Takes an eye image, facial anchor points, and target gaze angles,
    and produces a gaze-redirected eye image.
    
    Args:
        input_image: Eye image [batch, H, W, 3], normalized to [0, 1]
        anchor_points: Facial anchor point map [batch, H, W, anchor_dim]
        input_angles: Target gaze angles [batch, 2] (vertical, horizontal)
        is_training: Whether model is in training mode
        config: Model configuration
    
    Returns:
        Tuple of:
            - predicted_image: Gaze-corrected eye [batch, H, W, 3]
            - raw_flow: Optical flow before tanh [batch, H, W, 2]
            - lcm_weights: Light correction weights [batch, H, W, 2]
    
    Example:
        >>> config = ModelConfig()
        >>> with tf.Graph().as_default():
        ...     image_ph = tf.placeholder(tf.float32, [None, 48, 64, 3])
        ...     anchor_ph = tf.placeholder(tf.float32, [None, 48, 64, 12])
        ...     angle_ph = tf.placeholder(tf.float32, [None, 2])
        ...     pred, flow, lcm = build_inference_graph(
        ...         image_ph, anchor_ph, angle_ph, False, config
        ...     )
    """
    lcm_cfg = get_lcm_layer_config()
    
    with tf.compat.v1.variable_scope("warping_model"):
        # Encode gaze angles to spatial feature map
        angle_map = build_angle_encoder(
            input_angles, config.height, config.width, config.encoded_angle_dim
        )
        
        # Concatenate all inputs
        combined_inputs = tf.concat([input_image, anchor_points, angle_map], axis=3)
        
        # Build coarse-to-fine warping
        raw_flow, lcm_input = build_warping_module(
            combined_inputs, config, is_training
        )
        
        # Apply flow with tanh to bound displacement range
        bounded_flow = tf.nn.tanh(raw_flow)
        warped_image = spatial_transform.apply_optical_flow(
            bounded_flow, input_image, num_channels=3
        )
        
        # Apply light correction
        lcm_weights = build_lcm_module(lcm_input, lcm_cfg, is_training)
        predicted_image = apply_light_correction(warped_image, lcm_weights)
        
        return predicted_image, raw_flow, lcm_weights


################################################################################
# Loss Functions
################################################################################


def compute_image_loss(
    predicted: tf.Tensor,
    ground_truth: tf.Tensor,
    method: str = "MAE",
    border_crop: int = BORDER_CROP_PIXELS,
) -> tf.Tensor:
    """
    Compute image reconstruction loss with border cropping.
    
    Args:
        predicted: Predicted image [batch, H, W, C]
        ground_truth: Ground truth image [batch, H, W, C]
        method: 'MAE' for L1 loss, 'L2' for L2 loss
        border_crop: Pixels to ignore at borders
    
    Returns:
        Scalar loss value
    """
    with tf.compat.v1.variable_scope("image_reconstruction_loss"):
        if method == "L2":
            pixel_diff = tf.sqrt(
                tf.reduce_sum(tf.square(predicted - ground_truth), axis=3, keepdims=True)
            )
        else:  # MAE / L1
            pixel_diff = tf.abs(predicted - ground_truth)
        
        # Crop borders
        cropped = pixel_diff[:, border_crop:-border_crop, border_crop:-border_crop, :]
        
        # Sum over spatial and channel dimensions, average over batch
        per_sample_loss = tf.reduce_sum(cropped, axis=[1, 2, 3])
        return tf.reduce_mean(per_sample_loss)


def compute_total_variation(inputs: tf.Tensor) -> tf.Tensor:
    """
    Compute total variation of an image/flow field.
    
    TV measures the amount of "jumpiness" in an image, used as a
    regularization term to encourage smooth outputs.
    
    Args:
        inputs: Input tensor [batch, H, W, C]
    
    Returns:
        Per-pixel TV map [batch, H, W, 1]
    """
    with tf.compat.v1.variable_scope("total_variation"):
        # Horizontal gradients
        grad_x = inputs[:, :-1, :, :] - inputs[:, 1:, :, :]
        grad_x = tf.pad(grad_x, [[0, 0], [0, 1], [0, 0], [0, 0]], "CONSTANT")
        
        # Vertical gradients  
        grad_y = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]
        grad_y = tf.pad(grad_y, [[0, 0], [0, 0], [0, 1], [0, 0]], "CONSTANT")
        
        # Sum of absolute gradients
        total_var = tf.add(tf.abs(grad_x), tf.abs(grad_y))
        total_var = tf.reduce_sum(total_var, axis=3, keepdims=True)
        
        return total_var


def create_center_weight_map(
    shape: tf.Tensor,
    base_weight: float = 0.005,
    boundary_penalty: float = 3.0,
) -> tf.Tensor:
    """
    Create a weight map that increases toward image boundaries.
    
    Used to penalize TV loss more strongly at edges, encouraging
    the model to focus corrections in the center.
    
    Args:
        shape: Tensor shape [batch, height, width, channels]
        base_weight: Minimum weight at center
        boundary_penalty: Maximum weight at boundaries
    
    Returns:
        Weight map [batch, height, width, 1]
    """
    with tf.compat.v1.variable_scope("center_weight_map"):
        weight_range = boundary_penalty - base_weight
        
        # Create coordinate grids
        y_coords = tf.pow(tf.abs(tf.linspace(-1.0, 1.0, shape[1])), 8)
        x_coords = tf.pow(tf.abs(tf.linspace(-1.0, 1.0, shape[2])), 8)
        
        x_grid, y_grid = tf.meshgrid(x_coords, y_coords)
        x_grid = tf.expand_dims(x_grid, axis=2)
        y_grid = tf.expand_dims(y_grid, axis=2)
        
        # Distance from center
        coords = tf.concat([x_grid, y_grid], axis=2)
        distance = tf.sqrt(tf.reduce_sum(tf.square(coords), axis=2))
        
        # Scale and offset
        weights = weight_range * distance + base_weight
        
        # Add batch and channel dimensions
        weights = tf.expand_dims(
            tf.tile(tf.expand_dims(weights, axis=0), [shape[0], 1, 1]),
            axis=3
        )
        
        return weights


def compute_tv_losses(
    eye_mask: tf.Tensor,
    original_image: tf.Tensor,
    flow: tf.Tensor,
    lcm_weights: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute total variation losses for flow and LCM.
    
    Three separate TV terms:
    1. Eyeball TV: Penalize flow roughness in dark (eyeball) regions
    2. Eyelid TV: Penalize flow roughness in bright (eyelid) regions
    3. LCM TV: Penalize roughness in light correction weights
    
    Args:
        eye_mask: Binary mask for eye region [batch, H, W]
        original_image: Original eye image [batch, H, W, 3]
        flow: Optical flow [batch, H, W, 2]
        lcm_weights: LCM weights [batch, H, W, 2]
    
    Returns:
        Tuple of (eyeball_tv_loss, eyelid_tv_loss, lcm_tv_loss)
    """
    with tf.compat.v1.variable_scope("tv_losses"):
        flow_tv = compute_total_variation(flow)
        
        # Compute brightness-based weights (darker = more eyeball)
        grayscale = tf.reduce_mean(original_image, axis=3, keepdims=True)
        brightness_weight = 1.0 - grayscale
        
        # Expand eye mask
        eye_mask_4d = tf.expand_dims(eye_mask, axis=3)
        
        # Eyeball region: dark areas within eye mask
        eyeball_weights = tf.multiply(brightness_weight, eye_mask_4d)
        eyeball_tv = tf.multiply(eyeball_weights, flow_tv)
        
        # Eyelid region: outside eye mask
        eyelid_mask = 1.0 - eye_mask_4d
        eyelid_tv = tf.multiply(eyelid_mask, flow_tv)
        
        # Reduce to scalars
        eyeball_loss = tf.reduce_mean(tf.reduce_sum(eyeball_tv, axis=[1, 2, 3]))
        eyelid_loss = tf.reduce_mean(tf.reduce_sum(eyelid_tv, axis=[1, 2, 3]))
        
        # LCM TV with center weighting
        center_weights = create_center_weight_map(tf.shape(lcm_weights))
        lcm_tv = center_weights * compute_total_variation(lcm_weights)
        lcm_loss = tf.reduce_mean(tf.reduce_sum(lcm_tv, axis=[1, 2, 3]))
        
        return eyeball_loss, eyelid_loss, lcm_loss


def compute_lcm_regularization(lcm_weights: tf.Tensor) -> tf.Tensor:
    """
    Regularization loss for LCM weights.
    
    Penalizes the palette weight (second channel) to prevent
    excessive whitening of the output.
    
    Args:
        lcm_weights: LCM blending weights [batch, H, W, 2]
    
    Returns:
        Scalar regularization loss
    """
    with tf.compat.v1.variable_scope("lcm_regularization"):
        center_weights = create_center_weight_map(tf.shape(lcm_weights))
        
        # Extract palette weight channel
        _, palette_weight = tf.split(lcm_weights, [1, 1], axis=3)
        
        # Weighted penalty
        weighted_penalty = tf.abs(palette_weight) * center_weights
        per_sample = tf.reduce_sum(weighted_penalty, axis=[1, 2, 3])
        
        return tf.reduce_mean(per_sample)


def compute_total_loss(
    predicted_image: tf.Tensor,
    ground_truth_image: tf.Tensor,
    eye_mask: tf.Tensor,
    original_image: tf.Tensor,
    flow: tf.Tensor,
    lcm_weights: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute the complete training loss.
    
    Combines:
    - Image reconstruction loss
    - Eyeball TV loss
    - Eyelid TV loss
    - LCM TV loss
    - LCM regularization
    
    Args:
        predicted_image: Model output [batch, H, W, 3]
        ground_truth_image: Target image [batch, H, W, 3]
        eye_mask: Eye region mask [batch, H, W]
        original_image: Original input image [batch, H, W, 3]
        flow: Optical flow [batch, H, W, 2]
        lcm_weights: LCM weights [batch, H, W, 2]
    
    Returns:
        Tuple of (total_loss, image_loss)
    """
    with tf.compat.v1.variable_scope("losses"):
        image_loss = compute_image_loss(predicted_image, ground_truth_image, method="L2")
        
        eyeball_tv, eyelid_tv, lcm_tv = compute_tv_losses(
            eye_mask, original_image, flow, lcm_weights
        )
        
        lcm_reg = compute_lcm_regularization(lcm_weights)
        
        total = image_loss + eyeball_tv + eyelid_tv + lcm_tv + lcm_reg
        
        tf.add_to_collection("losses", total)
        total_loss = tf.add_n(tf.get_collection("losses"), name="total_loss")
        
        return total_loss, image_loss


################################################################################
# Backward Compatibility Aliases
################################################################################

# These functions maintain compatibility with existing code

def gen_agl_map(inputs, height, width, feature_dims):
    """Deprecated: Use tile_to_spatial_map instead."""
    return tile_to_spatial_map(inputs, height, width, feature_dims)

def encoder(inputs, height, width, tar_dim):
    """Deprecated: Use build_angle_encoder instead."""
    return build_angle_encoder(inputs, height, width, tar_dim)

def apply_lcm(batch_img, light_weight):
    """Deprecated: Use apply_light_correction instead."""
    return apply_light_correction(batch_img, light_weight)

def trans_module(inputs, structures, phase_train, name="trans_module"):
    """Deprecated: Use build_transform_module instead."""
    config = LayerConfig(
        depths=structures["depth"],
        kernel_sizes=structures["filter_size"],
    )
    return build_transform_module(inputs, config, phase_train, name)

def lcm_module(inputs, structures, phase_train, name="lcm_module"):
    """Deprecated: Use build_lcm_module instead."""
    config = LayerConfig(
        depths=structures["depth"],
        kernel_sizes=structures["filter_size"],
    )
    return build_lcm_module(inputs, config, phase_train, name)

def inference(input_img, input_fp, input_agl, phase_train, conf: ModelConfig):
    """Deprecated: Use build_inference_graph instead."""
    return build_inference_graph(input_img, input_fp, input_agl, phase_train, conf)

def dist_loss(y_pred, y_, method="MAE"):
    """Deprecated: Use compute_image_loss instead."""
    return compute_image_loss(y_pred, y_, method)

def TVloss(inputs):
    """Deprecated: Use compute_total_variation instead."""
    return compute_total_variation(inputs)

def TVlosses(eye_mask, ori_img, flow, lcm_map):
    """Deprecated: Use compute_tv_losses instead."""
    return compute_tv_losses(eye_mask, ori_img, flow, lcm_map)

def center_weight(shape, base=0.005, boundary_penalty=3.0):
    """Deprecated: Use create_center_weight_map instead."""
    return create_center_weight_map(shape, base, boundary_penalty)

def lcm_adj(lcm_wgt):
    """Deprecated: Use compute_lcm_regularization instead."""
    return compute_lcm_regularization(lcm_wgt)

def loss(img_pred, img_, eye_mask, input_img, flow, lcm_wgt):
    """Deprecated: Use compute_total_loss instead."""
    return compute_total_loss(img_pred, img_, eye_mask, input_img, flow, lcm_wgt)

# Global constant alias
img_crop = BORDER_CROP_PIXELS
