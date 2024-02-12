import torch
import tensorflow as tf
import numpy as np
import cv2

from violence_detection.constant import (
    MODEL_NAME_OR_PATH,
    INPUT_FRAMES,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    FPS,
)

GAMMA = 0.67  # if GAMMA < 0, no gamma correction will be applied
gamma_table = np.array(
    [((i / 255.0) ** GAMMA) * 255 for i in np.arange(0, 256)]
).astype("uint8")


class Inference:
    def __init__(
        self,
        model_path: str = MODEL_NAME_OR_PATH,
        fps=10,
        input_frame=INPUT_FRAMES,
        video_width=VIDEO_WIDTH,
        video_height=VIDEO_HEIGHT,
        sliding_step=FPS,
    ):
        self.fps = fps
        self.input_frame = input_frame
        self.video_width = video_width
        self.video_height = video_height
        self.sliding_step = sliding_step

        self.model_path = model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _autoloadmodel(self):
        from violence_detection.models.conv_lstm import model

        print("Loading model from ", self.model_path)
        self.model = tf.keras.models.load_model(self.model_path)

    def _predictimages(self, raw_video, skeleton_video, mean: bool = False):
        prob = self.model.predict([raw_video, skeleton_video])[0][0]
        prob = np.round(prob, 4)
        return prob

    def load_video(self, video_path) -> np.array:
        cap = cv2.VideoCapture(video_path)

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))

        if n_frames / original_fps < 5:
            print(f"[!] Video should be at least 5 seconds long")

        n_frames_to_keep = int(
            self.fps / original_fps * n_frames + 1
        )  # +1 is needed for frame_diff

        # Select frames equally spaced
        frames_idx = set(
            np.round(np.linspace(0, n_frames - 1, n_frames_to_keep)).astype(int)
        )

        frames = []
        index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if index in frames_idx:
                frame = cv2.resize(frame, (self.video_width, self.video_height)).astype(
                    np.float32
                )
                frames.append(frame)
            index += 1
        cap.release()

        return np.array(
            [
                frames[i : i + self.input_frame]
                for i in range(0, len(frames) - self.input_frame + 1, self.sliding_step)
            ]
        )
