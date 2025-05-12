import tensorflow as tf

################################################################################


def batch_norm(x, train_phase, name="bn_layer"):
    """
    Apply batch normalization with TF2.x compatibility that handles unknown shapes
    """
    batch_norm = tf.keras.layers.BatchNormalization(
        momentum=0.9,
        epsilon=1e-5,
        center=True,
        scale=True,
        name=name,
        trainable=True,  # Equivalent to TF1's 'training' parameter control
    )(x, training=train_phase)

    return batch_norm


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
