import tensorflow as tf
import tensorflow.keras.backend as K

from violence_detection.constant import WEIGHT_CURRENT


def tf_frame_diff(video):
    return video[1:] - video[:-1]


def tf_frame_dist(video):
    video_diff = tf_frame_diff(video)
    return K.sqrt(K.sum(K.square(video_diff), axis=-1, keepdims=True))


if WEIGHT_CURRENT:

    def tf_frame_diff_dist_combined(video):
        video_diff = tf_frame_diff(video)
        video_diff_current = tf.nn.relu(-video_diff)
        video_diff_next = tf.nn.relu(video_diff)
        video_diff_next_norm = K.sqrt(
            K.sum(K.square(video_diff_next), axis=-1, keepdims=True)
        )
        return K.concatenate([video_diff_current, video_diff_next_norm])

else:

    def tf_frame_diff_dist_combined(video):
        video_diff = tf_frame_diff(video)
        video_diff_current = tf.nn.relu(video_diff)
        video_diff_prev = tf.nn.relu(-video_diff)
        video_diff_prev_norm = K.sqrt(
            K.sum(K.square(video_diff_prev), axis=-1, keepdims=True)
        )
        return K.concatenate([video_diff_current, video_diff_prev_norm])
