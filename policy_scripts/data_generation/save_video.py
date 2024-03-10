import sys
sys.path.append('../')

import cv2
import numpy as np
from pathlib import Path
import pickle as pkl

from video import TrainVideoRecorder

# DATA_FOLDER = Path("/home/siddhant/github/retrieval/expert_demos/libero/libero_90")
DATA_FOLDER = Path("/home/siddhant/github/retrieval/single_robot/helper/demos/libero/libero_10")
names = ["KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"]

for name in names:
    DEMO_PATH = DATA_FOLDER / (name + ".pkl")

    with open(DEMO_PATH, "rb") as f:
        # demos, _, _, _, _ = pkl.load(f)
        demos, _, _, _ = pkl.load(f)

    SAVE_DIR = DATA_FOLDER
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    recorder = TrainVideoRecorder(SAVE_DIR)

    recorder.init(np.transpose(demos[0]['pixels'][0], (2, 0, 1)))
    for demo in demos[:5]:
        for frame in demo['pixels']:
            # cv2.imshow("frame", np.transpose(frame, (1, 2, 0)))
            # cv2.waitKey(5)
            recorder.record(np.transpose(frame, (2, 0, 1)))
    save_name = name + '.mp4'
    recorder.save(save_name)
