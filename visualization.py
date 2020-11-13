import random
from typing import List
from detectron2.utils.visualizer import Visualizer
from detectron2.data import Metadata
import cv2

#    dataset_dicts = get_balloon_dicts("balloon/train")


def visualize_dataset(dataset_dicts: List[dict], metadata: Metadata, num: int):
    out = []
    for d in random.sample(dataset_dicts, num):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        viz = visualizer.draw_dataset_dict(d)
        out.append(viz.get_image()[:, :, ::-1])
        # cv2_imshow(out.get_image()[:, :, ::-1])

    return out
