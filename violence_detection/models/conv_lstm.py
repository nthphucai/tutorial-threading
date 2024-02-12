import tensorflow as tf

from violence_detection.constant import (
    FRAMES_PER_VIDEO,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    N_CHANNELS,
    WEIGHT_CURRENT,
    FRAME_FUNC,
)
from violence_detection.models.keras.motion import (
    tf_frame_diff,
    tf_frame_dist,
    tf_frame_diff_dist_combined,
)

frame_func_dict = {
    "frame_diff": tf_frame_diff,
    "frame_dist": tf_frame_dist,
    "frame_diff_dist_combined": tf_frame_diff_dist_combined,
}

frame_func = frame_func_dict[FRAME_FUNC]

inputs_raw = tf.keras.layers.Input(
    shape=(FRAMES_PER_VIDEO, VIDEO_HEIGHT, VIDEO_WIDTH, N_CHANNELS)
)
inputs_openpose = tf.keras.layers.Input(
    shape=(FRAMES_PER_VIDEO, VIDEO_HEIGHT, VIDEO_WIDTH, N_CHANNELS)
)

inputs_diff = tf.keras.layers.Lambda(lambda video: tf.map_fn(frame_func, video))(
    inputs_raw
)

inputs_diff_norm = tf.keras.layers.BatchNormalization()(inputs_diff)
inputs_diff_time_info_weight = tf.keras.layers.ConvLSTM2D(
    filters=9,
    kernel_size=(3, 3),
    return_sequences=True,
    data_format="channels_last",
    activation="tanh",
)(inputs_diff_norm)


inputs_to_weight = inputs_openpose[:, :-1] if WEIGHT_CURRENT else inputs_openpose[:, 1:]

convolutional_layer = tf.keras.layers.Conv2D(
    filters=9, kernel_size=(3, 3), activation="relu"
)

inputs_openpose_soft = tf.keras.layers.TimeDistributed(convolutional_layer)(
    inputs_to_weight
)

inputs_openpose_norm = tf.keras.layers.BatchNormalization(scale=False, center=False)(
    inputs_openpose_soft
)

inputs_weighted = tf.keras.layers.Add()(
    [inputs_openpose_norm, inputs_diff_time_info_weight]
)

x = tf.keras.layers.ConvLSTM2D(
    filters=32,
    kernel_size=(3, 3),
    return_sequences=False,
    data_format="channels_last",
    activation="tanh",
)(inputs_weighted)

x = tf.keras.layers.DepthwiseConv2D(
    kernel_size=(3, 3),
    depth_multiplier=2,
    activation="relu",
    data_format="channels_last",
)(x)

x = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(x)

x = tf.keras.layers.Dense(units=128, activation="relu")(x)
x = tf.keras.layers.Dense(units=16, activation="relu")(x)
outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

model = tf.keras.Model([inputs_raw, inputs_openpose], outputs)


def run_test():
    import numpy as np

    # print(model.summary())

    raw_inpt = np.random.randn(4, 50, 100, 100, 3)
    skeleton_inpt = np.random.randn(4, 50, 100, 100, 3)

    out = model([raw_inpt, skeleton_inpt])

    print(out.shape)
