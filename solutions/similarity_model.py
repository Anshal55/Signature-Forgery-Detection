import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Activation,
    BatchNormalization,
    Input,
    Dropout,
    Flatten,
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import regularizers


def get_base_net_english(input_shape):
    """Base Siamese Network"""

    in_image = tf.keras.layers.Input(input_shape)

    c1 = tf.keras.layers.Conv2D(
        96,
        kernel_size=(11, 11),
        activation="relu",
        name="conv1_1",
        strides=4,
        kernel_initializer="glorot_uniform",
    )(in_image)
    # n1 = tf.keras.layers.BatchNormalization(epsilon=1e-06, momentum=0.9)(c1)
    n1 = tf.nn.local_response_normalization(
        c1, depth_radius=5, bias=1, alpha=0.0001, beta=0.75
    )
    p1 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(n1)
    zp1 = tf.keras.layers.ZeroPadding2D((2, 2))(p1)

    c2 = tf.keras.layers.Conv2D(
        256,
        kernel_size=(5, 5),
        activation="relu",
        name="conv2_1",
        strides=1,
        kernel_initializer="glorot_uniform",
    )(zp1)
    # n2 = tf.keras.layers.BatchNormalization(epsilon=1e-06, momentum=0.9)(c2)
    n2 = tf.nn.local_response_normalization(
        c2, depth_radius=5, bias=1, alpha=0.0001, beta=0.75
    )
    p2 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(n2)
    d1 = tf.keras.layers.Dropout(0.3)(p2)  # added extra
    zp2 = tf.keras.layers.ZeroPadding2D((1, 1))(d1)

    c3 = tf.keras.layers.Conv2D(
        384,
        kernel_size=(3, 3),
        activation="relu",
        name="conv3_1",
        strides=1,
        kernel_initializer="glorot_uniform",
    )(zp2)
    zp3 = tf.keras.layers.ZeroPadding2D((1, 1))(c3)

    c4 = tf.keras.layers.Conv2D(
        256,
        kernel_size=(3, 3),
        activation="relu",
        name="conv3_2",
        strides=1,
        kernel_initializer="glorot_uniform",
    )(zp3)
    p3 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(c4)
    d2 = tf.keras.layers.Dropout(0.3)(p3)  # added extra
    f1 = tf.keras.layers.Flatten(name="flatten")(d2)
    fc1 = tf.keras.layers.Dense(
        1024,
        kernel_regularizer=regularizers.l2(0.0005),
        activation="relu",
        kernel_initializer="glorot_uniform",
    )(f1)
    d3 = tf.keras.layers.Dropout(0.5)(fc1)

    out_embs = tf.keras.layers.Dense(
        128,
        kernel_regularizer=regularizers.l2(0.0005),
        activation="relu",
        kernel_initializer="glorot_uniform",
    )(
        d3
    )  # softmax changed to relu

    model = tf.keras.models.Model(inputs=in_image, outputs=out_embs)

    return model


def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1.0 - y_true) * margin_square)


def accuracy(y_true, y_pred):
    return tf.reduce_mean(
        tf.cast(tf.equal(y_true, tf.cast(y_pred < 0.5, y_true.dtype)), tf.float32)
    )


def get_main_model():
    input_a = Input(shape=(155, 220, 1))
    input_b = Input(shape=(155, 220, 1))

    base_net = get_base_net_english((155, 220, 1))
    processed_a = base_net(input_a)
    processed_b = base_net(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [processed_a, processed_b]
    )
    model = Model([input_a, input_b], distance)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4, rho=0.9, epsilon=1e-08)
    model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[accuracy])

    model.summary()
    return model
