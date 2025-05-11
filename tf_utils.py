import tensorflow as tf

# Dictionary to store BatchNormalization layers
_batch_norm_layers = {}

def batch_norm(x, train_phase, name="bn_layer"):
    """
    Apply batch normalization with TF2.x compatibility that handles unknown shapes
    """
    global _batch_norm_layers
    
    # First ensure we have a defined rank for the tensor
    if hasattr(x, 'shape') and x.shape.rank is None:
        # Force rank information if needed
        x = tf.ensure_shape(x, [None, None, None, None])  # Assuming 4D tensor for CNNs
    
    # Get the channel dimension (typically last dimension for conv layers)
    channel_dim = None
    if hasattr(x, 'shape') and x.shape.rank is not None and x.shape[-1] is not None:
        channel_dim = x.shape[-1]
    
    with tf.name_scope(name):
        # Create or reuse a BatchNormalization layer with explicit axis
        if name not in _batch_norm_layers:
            _batch_norm_layers[name] = tf.keras.layers.BatchNormalization(
                momentum=0.9,
                epsilon=1e-5,
                axis=-1,  # Always normalize over the channel dimension
                name=name
            )
        
        # Get the layer from the cache
        bn_layer = _batch_norm_layers[name]
        
        # Apply batch normalization with careful shape handling
        try:
            # Try the standard approach first
            return bn_layer(x, training=train_phase)
        except ValueError as e:
            # If that fails, try a more manual approach
            if "as_list" in str(e):
                # Create a simple manual batch normalization as fallback
                # This is a simplified version that mimics BatchNormalization for inference
                mean, variance = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
                return tf.nn.batch_normalization(
                    x, 
                    mean=mean, 
                    variance=variance, 
                    offset=None, 
                    scale=None, 
                    variance_epsilon=1e-5
                )
            else:
                # If it's another type of error, re-raise it
                raise


def cnn_blk(inputs, filters, kernel_size, phase_train, name="cnn_blk"):
    with tf.compat.v1.variable_scope(name) as scope:
        cnn = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation=None,
            use_bias=False,
            name="cnn",
        )(inputs)
        act = tf.nn.relu(cnn, name="act")
        ret = batch_norm(act, phase_train)
        return ret


def dnn_blk(inputs, nodes, name="dnn_blk"):
    with tf.compat.v1.variable_scope(name) as scope:
        dnn = tf.keras.layers.Dense(units=nodes, activation=None, name="dnn")(inputs)
        ret = tf.nn.relu(dnn, name="act")
        return ret
