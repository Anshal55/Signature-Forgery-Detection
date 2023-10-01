import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Activation,
    Input,
    Dropout,
    Flatten,
    Lambda,
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from solutions import model_metrics


def get_base_net_english(input_shape):
    """
    Base Siamese Network for English signatures.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).

    Returns:
        tf.keras.Model: The Siamese network model.
    """
    in_image = Input(input_shape)

    c1 = Conv2D(
        96,
        kernel_size=(11, 11),
        activation="relu",
        name="conv1_1",
        strides=4,
        kernel_initializer="glorot_uniform",
    )(in_image)
    n1 = tf.nn.local_response_normalization(
        c1, depth_radius=5, bias=1, alpha=0.0001, beta=0.75
    )
    p1 = MaxPooling2D((3, 3), strides=(2, 2))(n1)
    zp1 = tf.keras.layers.ZeroPadding2D((2, 2))(p1)

    c2 = Conv2D(
        256,
        kernel_size=(5, 5),
        activation="relu",
        name="conv2_1",
        strides=1,
        kernel_initializer="glorot_uniform",
    )(zp1)
    n2 = tf.nn.local_response_normalization(
        c2, depth_radius=5, bias=1, alpha=0.0001, beta=0.75
    )
    p2 = MaxPooling2D((3, 3), strides=(2, 2))(n2)
    d1 = Dropout(0.3)(p2)  # added extra
    zp2 = tf.keras.layers.ZeroPadding2D((1, 1))(d1)

    c3 = Conv2D(
        384,
        kernel_size=(3, 3),
        activation="relu",
        name="conv3_1",
        strides=1,
        kernel_initializer="glorot_uniform",
    )(zp2)
    zp3 = tf.keras.layers.ZeroPadding2D((1, 1))(c3)

    c4 = Conv2D(
        256,
        kernel_size=(3, 3),
        activation="relu",
        name="conv3_2",
        strides=1,
        kernel_initializer="glorot_uniform",
    )(zp3)
    p3 = MaxPooling2D((3, 3), strides=(2, 2))(c4)
    d2 = Dropout(0.3)(p3)  # added extra
    f1 = Flatten(name="flatten")(d2)
    fc1 = Dense(
        1024,
        kernel_regularizer=regularizers.l2(0.0005),
        activation="relu",
        kernel_initializer="glorot_uniform",
    )(f1)
    d3 = Dropout(0.5)(fc1)

    out_embs = Dense(
        128,
        kernel_regularizer=regularizers.l2(0.0005),
        activation="relu",
        kernel_initializer="glorot_uniform",
    )(d3)

    model = Model(inputs=in_image, outputs=out_embs)

    return model


def get_similarity_model():
    """
    Get the main similarity model.

    Returns:
        tf.keras.Model: The similarity model.
    """
    input_a = Input(shape=(155, 220, 1))
    input_b = Input(shape=(155, 220, 1))

    base_net = get_base_net_english((155, 220, 1))
    processed_a = base_net(input_a)
    processed_b = base_net(input_b)

    distance = Lambda(
        model_metrics.euclidean_distance,
        output_shape=model_metrics.eucl_dist_output_shape,
    )([processed_a, processed_b])
    model = Model([input_a, input_b], distance)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4, rho=0.9, epsilon=1e-08)
    model.compile(
        loss=model_metrics.contrastive_loss,
        optimizer=optimizer,
        metrics=[model_metrics.accuracy],
    )

    model.summary()
    return model
