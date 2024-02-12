import os
import sys

PARENT_PATH = os.getcwd()
sys.path.insert(1, os.path.join(PARENT_PATH, "violence_detection"))

import cv2
import threading
import datetime
import queue
import time
import argparse

import numpy as np

from inference import infer
from skeleton_extraction.extract_skeleton import SkeletonExtraction
from violence_detection.constant import FRAMES_PER_VIDEO

"""
Created on 12/2/2024

@Author: Nguyen Thi Hong Phuc 
Email: nthphucai@gmail.com 
"""


class ViolenceDetection:
    BATCH_SIZE = FRAMES_PER_VIDEO
    VIOLENCE_THRESH = 0.85
    isFightStatus = False
    isPlaying = True

    scoreFight = 0.0

    score_list = []

    image_queue = queue.Queue(400)
    image_condition = threading.Condition()

    pose_queue = queue.Queue(400)
    pose_condition = threading.Condition()

    draw_queue = queue.Queue(400)

    VIDEO_WIDTH = 100
    VIDEO_HEIGHT = 100
    FRAME_COUNT = 4

    GAMMA = 0.67  # if GAMMA < 0, no gamma correction will be applied
    FPS = 30

    def __init__(
        self,
        debug: bool = True,
        output_path: str = None,
        source_id_or_video_path: str = None,
    ):
        """Initialize the ViolenceDetection object.

        Args:
            debug (bool, optional): Enable debug mode (default is True).
            output_path (str, optional): Specify the output path for the result video (default is None).
            source_id_or_video_path (str, optional): Specify the source ID or video path for analysis (default is None).

        """
        self.debug = debug
        self.source_id_or_video_path = source_id_or_video_path
        self.output_path = output_path

        self.pose_model = SkeletonExtraction(yolo_model_path="models/yolov7-w6-pose.pt")

        self.violence_model = infer.Inference()
        self.violence_model._autoloadmodel()

        self.gamma_table = np.array(
            [((i / 255.0) ** self.GAMMA) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        self.thread1 = threading.Thread(
            target=self.camera_hanlder,
        )
        self.thread2 = threading.Thread(
            target=self.pose_handler,
        )
        self.thread3 = threading.Thread(
            target=self.violence_handler,
        )

        self.thread1.start()
        self.thread2.start()
        self.thread3.start()

    def stop_thread(self) -> None:
        """
        Stop the execution of the thread.

        This method sets the 'isPlaying' flag to False and notifies any waiting
        threads that are synchronized using the 'image_condition' and 'pose_condition'
        condition variables.

        :return: None
        """
        if self.debug:
            print("!!! STOP HERE !!!")

        self.isPlaying = False
        with self.image_condition:
            self.image_condition.notify_all()
        with self.pose_condition:
            self.pose_condition.notify_all()

    def pose_handler(self) -> None:
        """
        Handle pose estimation for frames in a continuous loop while the main thread is playing.

        This method continuously processes frames from the image queue, performs pose estimation using
        a pose model, and updates the pose queue and draw queue with relevant pose data. If the pose
        queue reaches the specified batch size, it notifies threads synchronized using the 'pose_condition'.
        Additionally, the draw queue is notified if it is not empty. Debug messages are printed if debugging
        is enabled.

        :return: None
        """
        frame_count = 0
        while self.isPlaying:
            try:
                with self.image_condition:
                    if self.image_queue.empty():
                        self.image_condition.wait()

                if self.debug:
                    print("run pose_handler ....", self.image_queue.qsize())

                if self.image_queue.qsize():
                    image_data = self.image_queue.get()
                    image = image_data["data"].copy()

                    # Increment frame count.
                    frame_count += 1
                    if frame_count % self.FRAME_COUNT != 0:
                        continue

                    frameClone = self.pose_model(image)
                    frameClone = cv2.LUT(frameClone, self.gamma_table)

                    pose_data = {
                        "skeleton_gamma": frameClone,
                        "raw_data": image_data["data"],
                        "time": image_data["time"],
                    }
                    self.pose_queue.put_nowait(pose_data)

                    if self.pose_queue.full():
                        self.pose_queue.get()

                    with self.pose_condition:
                        if self.pose_queue.qsize() > self.BATCH_SIZE:
                            self.pose_condition.notify_all()
                            if self.debug:
                                print("notify violence handler !!!")

                    self.draw_queue.put_nowait(pose_data)
                    if self.draw_queue.full():
                        self.draw_queue.get()

            except Exception as error:
                print("pose_handler errrrorrrrr ", type(error).__name__, "–", error)

    def violence_handler(self) -> None:
        """
        Handle violence detection based on pose data in a continuous loop while the main thread is playing.

        This method monitors the pose queue for a sufficient number of pose data entries (specified by 'BATCH_SIZE')
        before processing them for violence detection. Once the batch size is reached, it resizes and prepares the raw and
        gamma frames for analysis. The violence model predicts the probability of violence, and the 'scoreFight' attribute is
        updated accordingly. If the predicted score surpasses the specified violence threshold ('VIOLENCE_THRESH'), the
        'isFightStatus' attribute is set to True, indicating the presence of violence; otherwise, it is set to False.

        :return: None
        """
        while self.isPlaying:
            try:
                with self.pose_condition:
                    if self.pose_queue.qsize() < self.BATCH_SIZE:
                        if self.isPlaying:
                            self.pose_condition.wait()

                if self.debug:
                    print("run violence handler ....", self.pose_queue.qsize())

                if self.pose_queue.qsize() >= self.BATCH_SIZE:
                    raw_array = []
                    skeleton_array = []
                    for _ in range(0, self.BATCH_SIZE):
                        pose_data = self.pose_queue.get()

                        raw_frame = cv2.resize(
                            pose_data["raw_data"], (self.VIDEO_WIDTH, self.VIDEO_HEIGHT)
                        ).astype(np.float32)
                        raw_array.append(raw_frame)

                        skeleton_frame = cv2.resize(
                            pose_data["skeleton_gamma"],
                            (self.VIDEO_WIDTH, self.VIDEO_HEIGHT),
                        ).astype(np.float32)
                        skeleton_array.append(skeleton_frame)

                    numpy_raw = np.array(raw_array)
                    numpy_raw = numpy_raw[np.newaxis]

                    numpy_skeleton = np.array(skeleton_array)
                    numpy_skeleton = numpy_skeleton[np.newaxis]

                    self.scoreFight = self.violence_model._predictimages(
                        numpy_raw, numpy_skeleton
                    )
                    print("probability", self.scoreFight)

                    if self.scoreFight > self.VIOLENCE_THRESH:
                        self.isFightStatus = True
                    else:
                        self.isFightStatus = False

            except Exception as error:
                print("violence_handler errrrorrrrr ", type(error).__name__, "–", error)

    def camera_hanlder(self):
        """
        Handles the camera for capturing and processing video frames.

        This method configures the video capture based on the provided source ID or video path.
        If the source is a string (video file path), it sets up the video capture properties
        such as frame width, height, FPS, and codec. If the source is an integer (camera ID),
        it configures properties like frame width, height, FPS, and codec accordingly.

        Parameters:
            self.source_id_or_video_path (str or int): Source ID or video path for camera input.
            self.output_path (str): Output path for the processed video.
            self.FPS (int): Frames per second for video capture (default: None).
            self.debug (bool): Enable debug mode (default: False).

        Raises:
            NotImplementedError: If the source type is not recognized.

        Returns:
            None

        Note:
            The method continuously captures video frames, processes them, and displays the result.
            Press 'q' to exit the camera handler.

        """

        if isinstance(self.source_id_or_video_path, str):
            vid = cv2.VideoCapture(self.source_id_or_video_path)
            self.fps = (
                int(vid.get(cv2.CAP_PROP_FPS)) if self.FPS is not None else self.FPS
            )
            self.width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fourcc = cv2.VideoWriter_fourcc(*"XVID")

        elif isinstance(self.source_id_or_video_path, int):
            vid = cv2.VideoCapture(self.source_id_or_video_path)
            self.fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
            vid.set(cv2.CAP_PROP_FOURCC, self.fourcc)
            self.width = 640
            self.height = 360
            self.fps = 30 if self.FPS is not None else self.FPS
            vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            vid.set(cv2.CAP_PROP_FPS, self.fps)
        else:
            raise NotImplementedError("Please check camera source input!!!")

        self.size = (self.width, self.height)
        out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, self.size)
        print("fourcc:", self.fourcc, "fps:", self.fps, "size:", self.size)

        frame_count = 0
        ptime = 0
        ctime = 0
        while self.isPlaying:
            try:
                # Capture the video frame
                # by frame
                ret, frame = vid.read()
                if not ret:
                    break

                frame_count += 1
                # print("frame_count:", frame_count)

                curTime = time.time()
                frame_data = {"data": frame, "time": curTime}

                self.image_queue.put_nowait(frame_data)

                if self.image_queue.full():
                    self.image_queue.get()

                with self.image_condition:
                    if not self.image_queue.empty():
                        self.image_condition.notify_all()
                        if self.debug:
                            print("notify pose_handler !!!")

                if self.draw_queue.qsize():
                    draw_data = self.draw_queue.get()
                    #skeleton_gamma = draw_data["skeleton_gamma"]
                    skeleton_gamma = draw_data["raw_data"]

                    ctime = time.time()
                    fps = 1 / (ctime - ptime)
                    ptime = ctime

                    font = cv2.FONT_HERSHEY_PLAIN
                    dt = (
                        " FPS: "
                        + str(int(fps))
                        + " "
                        + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        + " "
                        + str(self.isFightStatus)
                        + " "
                        + str(self.scoreFight)
                    )

                    skeleton_gamma = cv2.putText(
                        skeleton_gamma,
                        dt,
                        (10, 20),
                        font,
                        1,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    out.write(skeleton_gamma)
                    cv2.imshow("camera", skeleton_gamma)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except Exception as error:
                print("camera_handler errrrorrrrr ", type(error).__name__, "–", error)

        self.stop_thread()
        vid.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multithreading for Violence Detection"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_video.avi",
        help="Output path for the result video",
    )
    parser.add_argument(
        "--source_id_or_video_path",
        type=str,
        default="test_video.avi",
        help="Source ID or video path for analysis",
    )
    # parser.add_argument("--source_id_or_video_path", type=str, default=0, help="Source ID or video path for analysis")

    args = parser.parse_args()

    ViolenceDetection(
        debug=args.debug,
        output_path=args.output_path,
        source_id_or_video_path=args.source_id_or_video_path,
    )
