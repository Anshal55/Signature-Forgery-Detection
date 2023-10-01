import tensorflow.keras.backend as K


def euclidean_distance(vects):
    """
    Compute the Euclidean distance between two vectors.

    Args:
        vects (list): A list containing two vectors.

    Returns:
        tf.Tensor: The Euclidean distance as a tensor.
    """
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    """
    Define the output shape of the Euclidean distance layer.

    Args:
        shapes (tuple): A tuple of input shapes.

    Returns:
        tuple: The output shape tuple.
    """
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss function.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.

    Returns:
        tf.Tensor: The contrastive loss as a tensor.
    """
    margin = 1.0
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1.0 - y_true) * margin_square)


def accuracy(y_true, y_pred):
    """
    Compute accuracy.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.

    Returns:
        tf.Tensor: The accuracy as a tensor.
    """
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
