"""
Spatial Transformation Module for Gaze Correction

This module provides differentiable spatial transformation operations
that warp images based on optical flow fields. It's used to apply
learned deformations to eye images for gaze redirection.

Key concepts:
- Flow field: Per-pixel displacement vectors (dx, dy)
- Bilinear interpolation: Smooth sampling at non-integer coordinates
- Meshgrid: Coordinate system for pixel locations

The transformation is differentiable, enabling end-to-end training.
"""

import tensorflow as tf
from typing import Tuple


################################################################################
# Coordinate Utilities
################################################################################


def create_meshgrid(height: int, width: int) -> tf.Tensor:
    """
    Create a normalized coordinate grid for an image.
    
    Creates a grid of (x, y) coordinates normalized to [-1, 1] range,
    where (-1, -1) is top-left and (1, 1) is bottom-right.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
    
    Returns:
        Coordinate grid tensor of shape [2, height * width]
        - Row 0: x coordinates (horizontal)
        - Row 1: y coordinates (vertical)
    
    Example:
        >>> grid = create_meshgrid(48, 64)
        >>> # grid[0] contains 48*64 x-coordinates in [-1, 1]
        >>> # grid[1] contains 48*64 y-coordinates in [-1, 1]
    """
    with tf.name_scope("create_meshgrid"):
        # Create linearly spaced coordinates
        y_coords = tf.linspace(-1.0, 1.0, height)
        x_coords = tf.linspace(-1.0, 1.0, width)
        
        # Create 2D grid
        x_grid, y_grid = tf.meshgrid(x_coords, y_coords)
        
        # Flatten to [1, num_pixels] and concatenate
        y_flat = tf.expand_dims(tf.reshape(y_grid, [-1]), 0)
        x_flat = tf.expand_dims(tf.reshape(x_grid, [-1]), 0)
        
        return tf.concat([x_flat, y_flat], axis=0)


def repeat_vector(vector: tf.Tensor, num_repeats: int) -> tf.Tensor:
    """
    Repeat each element of a vector multiple times.
    
    This is used to expand batch indices when gathering pixels.
    
    Args:
        vector: 1D tensor to repeat
        num_repeats: Number of times to repeat each element
    
    Returns:
        Repeated tensor with length = len(vector) * num_repeats
    
    Example:
        >>> v = tf.constant([0, 1, 2])
        >>> repeat_vector(v, 3)
        >>> # Returns: [0, 0, 0, 1, 1, 1, 2, 2, 2]
    """
    with tf.name_scope("repeat_vector"):
        ones_matrix = tf.ones((1, num_repeats), dtype=tf.int32)
        column_vector = tf.reshape(vector, shape=(-1, 1))
        repeated = tf.matmul(column_vector, ones_matrix)
        return tf.reshape(repeated, [-1])


################################################################################
# Bilinear Interpolation
################################################################################


def bilinear_interpolate(
    image: tf.Tensor,
    sample_x: tf.Tensor,
    sample_y: tf.Tensor,
    output_size: Tuple[int, int],
) -> tf.Tensor:
    """
    Sample from an image using bilinear interpolation.
    
    Given an image and sampling coordinates, compute output pixel values
    by interpolating between the four nearest neighbors of each sample point.
    
    This operation is differentiable, enabling gradient flow through
    the sampling coordinates for end-to-end learning.
    
    Args:
        image: Source image tensor [batch, height, width, channels]
        sample_x: X coordinates to sample, normalized to [-1, 1]
                  Flattened to [batch * output_height * output_width]
        sample_y: Y coordinates to sample, normalized to [-1, 1]
                  Flattened to [batch * output_height * output_width]
        output_size: (output_height, output_width) of result
    
    Returns:
        Sampled image [batch, output_height, output_width, channels]
    
    Coordinate system:
        - (-1, -1): top-left corner
        - ( 1,  1): bottom-right corner
        - ( 0,  0): image center
    """
    with tf.name_scope("bilinear_interpolate"):
        # Get image dimensions
        batch_size, height, width, num_channels = tf.unstack(tf.shape(image))
        output_height, output_width = output_size
        
        # Convert normalized coordinates to pixel coordinates
        sample_x = tf.cast(sample_x, tf.float32)
        sample_y = tf.cast(sample_y, tf.float32)
        height_float = tf.cast(height, tf.float32)
        width_float = tf.cast(width, tf.float32)
        
        # Map from [-1, 1] to [0, width/height]
        pixel_x = 0.5 * (sample_x + 1.0) * width_float
        pixel_y = 0.5 * (sample_y + 1.0) * height_float
        
        # Find corner coordinates of the 4 neighbors
        x_floor = tf.cast(tf.floor(pixel_x), tf.int32)
        x_ceil = x_floor + 1
        y_floor = tf.cast(tf.floor(pixel_y), tf.int32)
        y_ceil = y_floor + 1
        
        # Clamp coordinates to valid range
        max_x = width - 1
        max_y = height - 1
        x_floor = tf.clip_by_value(x_floor, 0, max_x)
        x_ceil = tf.clip_by_value(x_ceil, 0, max_x)
        y_floor = tf.clip_by_value(y_floor, 0, max_y)
        y_ceil = tf.clip_by_value(y_ceil, 0, max_y)
        
        # Compute flat indices for gathering pixels
        pixels_per_image = height * width
        num_output_pixels = output_height * output_width
        
        # Offset for each batch item
        batch_offsets = tf.range(batch_size) * pixels_per_image
        base_indices = repeat_vector(batch_offsets, num_output_pixels)
        
        # Indices for the four corners: top-left, bottom-left, top-right, bottom-right
        base_y_floor = base_indices + y_floor * width
        base_y_ceil = base_indices + y_ceil * width
        
        indices_top_left = base_y_floor + x_floor
        indices_bottom_left = base_y_ceil + x_floor
        indices_top_right = base_y_floor + x_ceil
        indices_bottom_right = base_y_ceil + x_ceil
        
        # Gather pixel values from flattened image
        flat_image = tf.reshape(image, (-1, num_channels))
        flat_image = tf.cast(flat_image, tf.float32)
        
        pixels_top_left = tf.gather(flat_image, indices_top_left)
        pixels_bottom_left = tf.gather(flat_image, indices_bottom_left)
        pixels_top_right = tf.gather(flat_image, indices_top_right)
        pixels_bottom_right = tf.gather(flat_image, indices_bottom_right)
        
        # Compute interpolation weights (areas of opposite rectangles)
        x_floor_f = tf.cast(x_floor, tf.float32)
        x_ceil_f = tf.cast(x_ceil, tf.float32)
        y_floor_f = tf.cast(y_floor, tf.float32)
        y_ceil_f = tf.cast(y_ceil, tf.float32)
        
        weight_top_left = tf.expand_dims((x_ceil_f - pixel_x) * (y_ceil_f - pixel_y), 1)
        weight_bottom_left = tf.expand_dims((x_ceil_f - pixel_x) * (pixel_y - y_floor_f), 1)
        weight_top_right = tf.expand_dims((pixel_x - x_floor_f) * (y_ceil_f - pixel_y), 1)
        weight_bottom_right = tf.expand_dims((pixel_x - x_floor_f) * (pixel_y - y_floor_f), 1)
        
        # Weighted sum of four corners
        output = tf.add_n([
            weight_top_left * pixels_top_left,
            weight_bottom_left * pixels_bottom_left,
            weight_top_right * pixels_top_right,
            weight_bottom_right * pixels_bottom_right,
        ])
        
        return output


################################################################################
# Flow-Based Transformation
################################################################################


def apply_optical_flow(
    flow_field: tf.Tensor,
    source_image: tf.Tensor,
    num_channels: int = 3,
) -> tf.Tensor:
    """
    Warp an image using an optical flow field.
    
    For each output pixel, the flow field specifies where to sample
    from the source image. This implements backward warping:
    output[y, x] = source[y + flow_y[y,x], x + flow_x[y,x]]
    
    Args:
        flow_field: Per-pixel displacements [batch, height, width, 2]
                    Channel 0: horizontal displacement (dx)
                    Channel 1: vertical displacement (dy)
                    Values are normalized to [-1, 1]
        source_image: Image to warp [batch, height, width, channels]
        num_channels: Number of image channels (default: 3 for RGB)
    
    Returns:
        Warped image [batch, height, width, channels]
    
    Example:
        >>> flow = model.predict_flow(eye_image, gaze_angle)
        >>> warped_eye = apply_optical_flow(flow, eye_image)
    
    Notes:
        - Flow of (0, 0) means no displacement (identity transform)
        - Positive dx shifts sampling to the right
        - Positive dy shifts sampling downward
    """
    with tf.name_scope("apply_optical_flow"):
        batch_size = tf.shape(source_image)[0]
        height = tf.shape(source_image)[1]
        width = tf.shape(source_image)[2]
        output_size = (height, width)
        
        num_flow_channels = tf.shape(flow_field)[3]
        
        # Reshape flow to [batch, 2, height * width]
        flow_transposed = tf.transpose(flow_field, [0, 3, 1, 2])
        flow_flat = tf.reshape(
            flow_transposed,
            [batch_size, num_flow_channels, height * width]
        )
        
        # Create base coordinate grid
        base_grid = create_meshgrid(height, width)
        
        # Add flow to base coordinates
        transformed_coords = flow_flat + base_grid
        
        # Extract x and y sample coordinates
        sample_x = tf.slice(transformed_coords, [0, 0, 0], [-1, 1, -1])
        sample_y = tf.slice(transformed_coords, [0, 1, 0], [-1, 1, -1])
        
        sample_x_flat = tf.reshape(sample_x, [-1])
        sample_y_flat = tf.reshape(sample_y, [-1])
        
        # Sample source image at transformed coordinates
        warped_flat = bilinear_interpolate(
            source_image, sample_x_flat, sample_y_flat, (height, width)
        )
        
        # Reshape to output dimensions
        warped_image = tf.reshape(
            warped_flat,
            [batch_size, height, width, num_channels]
        )
        
        return warped_image


################################################################################
# Backward Compatibility
################################################################################

# Alias for backward compatibility with existing code
def apply_transformation(flows, img, num_channels=3):
    """
    Deprecated: Use apply_optical_flow instead.
    
    This function is kept for backward compatibility.
    """
    return apply_optical_flow(flows, img, num_channels)


# Additional backward-compatible names
meshgrid = create_meshgrid
repeat = repeat_vector
interpolate = bilinear_interpolate
