import sys

sys.path.append("skeleton_extraction")

import cv2
import numpy as np

import torch
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.system_utils import get_progress


class SkeletonExtraction:
    GAMMA = 0.67

    def __init__(self, yolo_model_path: str = "yolov7-w6-pose.pt") -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        weigths = torch.load(yolo_model_path, map_location=self.device)
        self.model = weigths["model"]

        self.model = self.model.half().to(self.device)
        _ = self.model.eval()

    def __call__(self, frame):
        frame_height, frame_width, _ = frame.shape

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = letterbox(frame, (frame_width), stride=64, auto=True)[0]
        frame = transforms.ToTensor()(frame)
        frame = torch.tensor(np.array([frame.numpy()]))
        frame = frame.to(self.device)
        frame = frame.half()

        with torch.no_grad():
            output, _ = self.model(frame)

        frame = self._draw_keypoints(output, frame)
        frame = cv2.resize(frame, (frame_width, frame_height))
        return frame

    def _draw_keypoints(self, output, image):
        output = non_max_suppression_kpt(
            output,
            0.25,  # Confidence Threshold
            0.65,  # IoU Threshold
            nc=self.model.yaml["nc"],  # Number of Classes
            nkpt=self.model.yaml["nkpt"],  # Number of Keypoints
            kpt_label=True,
        )
        with torch.no_grad():
            output = output_to_keypoint(output)
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

        frameClone = np.zeros_like(nimg)
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(frameClone, output[idx, 7:].T, 3)

        return frameClone
