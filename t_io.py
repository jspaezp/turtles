import pandas as pd
from pathlib import Path
import cv2
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog


def get_turtle_dicts(img_dir: Path):
    csv_file = Path(img_dir) / "entries.csv"
    img_annotations = pd.read_csv(str(csv_file))

    dataset_dicts = []

    for idx, row in enumerate(img_annotations.iterrows()):
        record = {}
        image_id = row[1]["Image_ID"]
        filename = Path(img_dir) / f"{image_id}.JPG"

        assert filename.is_file()

        height, width = cv2.imread(str(filename)).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        bbox = [
            row[1]["x"] * width,
            row[1]["y"] * height,
            (row[1]["x"] + row[1]["w"]) * width,
            (row[1]["y"] + row[1]["h"]) * height,
        ]

        obj = {
            "bbox": bbox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 0,
        }

        record["annotations"] = [obj]
        dataset_dicts.append(record)
    return dataset_dicts


def register_datasets(split_dir: Path):
    for d in ["train", "test"]:
        entry_name = "turtle_" + d
        DatasetCatalog.register(entry_name, lambda d=d: get_turtle_dicts(split_dir / d))
        MetadataCatalog.get(entry_name).set(thing_classes=["turtle"])
        print(f"Registered {entry_name} to the dataset catalog")

    turtle_metadata = MetadataCatalog.get("turtle_train")
    return turtle_metadata
