DATASET_NAME = "INCAR"

BACKGROUND = False  # Use output of OpenPose with or without background

WEIGHT_CURRENT = True

FRAME_FUNC = "frame_diff"

CONFIG_FILE = "datasets/INCAR/InCar_GT_annotations.json"

FULL_DATA_PATH = f"datasets/{DATASET_NAME}/full_data"

# Paths to videos for training
PATHS = [
    f"{FULL_DATA_PATH}/original_data",
    f"{FULL_DATA_PATH}/openpose_gamma",
]

FACTOR_NUM = 50

ORIGINAL_FRAMES_PER_VIDEO = 100
FRAMES_PER_VIDEO = FACTOR_NUM + 1

VIDEO_WIDTH, VIDEO_HEIGHT = 100, 100
N_CHANNELS = 3
FPS = 10

INPUT_FRAMES = FACTOR_NUM + 1  # number of frames that the model receives
SLIDING_STEP = FACTOR_NUM + 1

YOLO_MODEL_PATH = "./models/yolov7-w6-pose.pt"

MODEL_NAME_OR_PATH = "./models/full_data_frame_diff_gamma_best_model.h5"
