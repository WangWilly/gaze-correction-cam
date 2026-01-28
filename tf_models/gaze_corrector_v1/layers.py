"""
TensorFlow Neural Network Layers Module

This module provides reusable neural network layer building blocks
for the gaze correction model, including:
- Batch normalization
- Convolutional blocks
- Dense (fully-connected) blocks

These are designed to be compatible with TensorFlow 1.x style graphs
while using TensorFlow 2.x APIs where possible.
"""

import tensorflow as tf
from typing import Union, List


################################################################################
# Normalization Layers
################################################################################


def batch_normalization(
    inputs: tf.Tensor,
    is_training: bool,
    name: str = "batch_norm",
    momentum: float = 0.9,
    epsilon: float = 1e-5,
) -> tf.Tensor:
    """
    Apply batch normalization to inputs.
    
    Batch normalization helps stabilize training by normalizing layer inputs
    to have zero mean and unit variance, then applying learnable scale and shift.
    
    Args:
        inputs: Input tensor of any shape
        is_training: Whether the model is in training mode
            - True: Use batch statistics for normalization
            - False: Use running mean/variance for inference
        name: Name scope for the layer
        momentum: Momentum for the moving average (default: 0.9)
        epsilon: Small constant to prevent division by zero (default: 1e-5)
    
    Returns:
        Normalized tensor with same shape as input
    
    Example:
        >>> x = tf.random.normal([32, 48, 64, 32])
        >>> normalized = batch_normalization(x, is_training=True)
    """
    return tf.keras.layers.BatchNormalization(
        momentum=momentum,
        epsilon=epsilon,
        center=True,
        scale=True,
        name=name,
        trainable=True,
    )(inputs, training=is_training)


################################################################################
# Convolutional Layers
################################################################################


def conv2d_layer(
    inputs: tf.Tensor,
    filters: int,
    kernel_size: Union[int, List[int]],
    activation: str = None,
    use_bias: bool = False,
    padding: str = "same",
    name: str = "conv2d",
) -> tf.Tensor:
    """
    Create a 2D convolutional layer.
    
    Args:
        inputs: Input tensor with shape [batch, height, width, channels]
        filters: Number of output feature maps (filter count)
        kernel_size: Size of convolution kernel [height, width] or single int
        activation: Activation function name ('relu', 'tanh', etc.) or None
        use_bias: Whether to add bias to outputs
        padding: Padding mode - 'same' preserves spatial dimensions
        name: Name for the layer
    
    Returns:
        Convolved tensor with shape [batch, height, width, filters]
    """
    return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
        use_bias=use_bias,
        name=name,
    )(inputs)


def conv_block(
    inputs: tf.Tensor,
    filters: int,
    kernel_size: Union[int, List[int]],
    is_training: bool,
    name: str = "conv_block",
) -> tf.Tensor:
    """
    Convolutional block: Conv2D -> ReLU -> BatchNorm.
    
    A commonly used pattern in deep networks that applies:
    1. Convolution to extract features
    2. ReLU activation for non-linearity
    3. Batch normalization for training stability
    
    Args:
        inputs: Input tensor [batch, height, width, channels]
        filters: Number of output feature maps
        kernel_size: Convolution kernel size [h, w]
        is_training: Whether model is in training mode
        name: Name scope for the block
    
    Returns:
        Processed tensor [batch, height, width, filters]
    
    Example:
        >>> x = tf.random.normal([32, 48, 64, 3])
        >>> features = conv_block(x, filters=32, kernel_size=[3, 3], is_training=True)
    """
    with tf.compat.v1.variable_scope(name):
        # Use "cnn" for checkpoint compatibility
        conv_output = conv2d_layer(
            inputs,
            filters=filters,
            kernel_size=kernel_size,
            activation=None,
            use_bias=False,
            name="cnn",
        )
        # Use "act" for checkpoint compatibility
        activated = tf.nn.relu(conv_output, name="act")
        # Use "bn_layer" for checkpoint compatibility
        normalized = batch_normalization(activated, is_training, name="bn_layer")
        return normalized


################################################################################
# Dense (Fully-Connected) Layers
################################################################################


def dense_layer(
    inputs: tf.Tensor,
    units: int,
    activation: str = None,
    use_bias: bool = True,
    name: str = "dense",
) -> tf.Tensor:
    """
    Create a fully-connected (dense) layer.
    
    Args:
        inputs: Input tensor, typically [batch, features]
        units: Number of output neurons
        activation: Activation function name or None
        use_bias: Whether to add bias term
        name: Name for the layer
    
    Returns:
        Dense layer output [batch, units]
    """
    return tf.keras.layers.Dense(
        units=units,
        activation=activation,
        use_bias=use_bias,
        name=name,
    )(inputs)


def dense_block(
    inputs: tf.Tensor,
    units: int,
    name: str = "dense_block",
) -> tf.Tensor:
    """
    Dense block: Dense -> ReLU.
    
    A simple pattern for fully-connected layers with ReLU activation.
    
    Args:
        inputs: Input tensor [batch, features]
        units: Number of output neurons
        name: Name scope for the block
    
    Returns:
        Processed tensor [batch, units]
    
    Example:
        >>> x = tf.random.normal([32, 128])
        >>> out = dense_block(x, units=64, name="fc_layer")
    """
    with tf.compat.v1.variable_scope(name):
        # Use "dnn" for checkpoint compatibility
        dense_output = dense_layer(inputs, units=units, activation=None, name="dnn")
        # Use "act" for checkpoint compatibility
        activated = tf.nn.relu(dense_output, name="act")
        return activated


################################################################################
# Pooling Layers
################################################################################


def average_pooling_2d(
    inputs: tf.Tensor,
    pool_size: tuple = (2, 2),
    strides: tuple = (2, 2),
    padding: str = "same",
    name: str = "avg_pool",
) -> tf.Tensor:
    """
    Apply 2D average pooling.
    
    Args:
        inputs: Input tensor [batch, height, width, channels]
        pool_size: Size of the pooling window (height, width)
        strides: Step size for pooling window (height, width)
        padding: 'same' or 'valid'
        name: Name for the layer
    
    Returns:
        Pooled tensor with reduced spatial dimensions
    """
    return tf.keras.layers.AveragePooling2D(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        name=name,
    )(inputs)


################################################################################
# Backward Compatibility Aliases
################################################################################

# These aliases maintain compatibility with existing code that uses old names
cnn_blk = conv_block
dnn_blk = dense_block
batch_norm = batch_normalization
