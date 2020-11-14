from pathlib import Path
import pandas as pd
import numpy as np
from detectron2.engine import DefaultPredictor
import cv2
from progress.bar import Bar
import train
import click
from visualization import visualize_predictions
from detectron2.config import CfgNode
from collections import namedtuple


def get_predictor(cfg=train.config_model(), model_name="model_final.pth"):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    model_path = Path(cfg.OUTPUT_DIR) / model_name  # path to the model we just trained
    assert model_path.is_file()

    cfg.MODEL.WEIGHTS = str(model_path)

    cfg.RETINANET.SCORE_THRESH_TEST = 0.7  # used for retinanet
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        0.7  # set a custom testing threshold, used for maskrcnn
    )
    predictor = DefaultPredictor(cfg)
    return predictor


def predict_submission(sub_file: Path, predictor: DefaultPredictor, img_path: Path):
    assert img_path.is_dir()
    df = pd.read_csv(str(sub_file))
    outs = []

    # TODO: this can probably be optimized passing a tensor and not
    # in a for loop ...

    bar = Bar("Processing", max=len(df))

    for base_filename in df["Image_ID"]:
        file_name = img_path / f"{base_filename}.JPG"
        assert file_name.is_file()

        im = cv2.imread(str(file_name))
        outputs = predictor(im)

        pred_df = prediction_to_df(outputs)
        pred_df["Image_ID"] = base_filename

        outs.append(pred_df)
        bar.next()

    bar.finish()
    out_df = pd.concat(outs, axis=0)

    # Keep only the highest scoring hit
    top_hit_df = get_top_hits(out_df)

    # This section takes care of ordering the output df in the same
    # order as the original submission df
    top_hit_df = top_hit_df.set_index("Image_ID").reindex(
        columns=["Image_ID", "x", "y", "w", "h", "score"]
    )

    top_hit_df = top_hit_df.loc[df["Image_ID"]]
    del top_hit_df["Image_ID"]

    top_hit_df = top_hit_df.reset_index()

    return top_hit_df


def prediction_to_df(outputs):
    pred = outputs["instances"].to("cpu")
    # height (y), width (x)
    img_size = pred.__dict__["_image_size"]
    box_scores = pred.__dict__["_fields"]["scores"].numpy()
    #  Each row is (x1, y1, x2, y2).
    boxes = pred.__dict__["_fields"]["pred_boxes"].__dict__["tensor"].numpy()

    out_boxes = boxes.copy()

    # Convert to relative measurements
    out_boxes[:, 0::2] = out_boxes[:, 0::2] / img_size[1]
    out_boxes[:, 1::2] = out_boxes[:, 1::2] / img_size[0]

    # Sub entries have to be x, y, w, h
    out_boxes[:, 2] = out_boxes[:, 2] - out_boxes[:, 0]
    out_boxes[:, 3] = out_boxes[:, 3] - out_boxes[:, 1]

    df = pd.DataFrame(out_boxes, columns=["x", "y", "w", "h"])
    df["score"] = box_scores
    return df


Prediction = col.namedtuple(
    "Prediction", ["prediction_output", "prediction_visualization"]
)


def predict_file(img: Path, predictor: DefaultPredictor, config: CfgNode) -> Prediction:
    im = cv2.imread(str(img))
    outputs = predictor(im)
    viz = visualize_predictions(im, outputs, config)
    return Prediction(outputs, viz)


def get_top_hits(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the highest scores for each Image_ID

    :param df: a DataFrame with the predictions of a model, generated with prediction_to_df
    :type df: pd.DataFrame
    :return: Dataframe keeping only rows that have the highest score for each Image_ID
    :rtype: pd.DataFrame
    """
    df_ui = df.reset_index()
    idmaxes = df_ui.groupby(["Image_ID"])["score"].idxmax()
    top_hit_df = df_ui.loc[idmaxes]
    top_hit_df = top_hit_df.reset_index()

    return top_hit_df


def pandas_iou(joint_df):
    # boxA is actually the predictions - a list of [[x, y, w, h], ...]
    # boxB is the targets, same format

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = pd.concat([joint_df["x"], joint_df["x_pred"]], axis=1).max(axis=1)
    yA = pd.concat([joint_df["y"], joint_df["y_pred"]], axis=1).max(axis=1)

    xB = pd.concat(
        [joint_df["x"] + joint_df["w"], joint_df["x_pred"] + joint_df["w_pred"]], axis=1
    ).min(
        axis=1
    )  # x+w
    yB = pd.concat(
        [joint_df["y"] + joint_df["h"], joint_df["y_pred"] + joint_df["h_pred"]], axis=1
    ).min(
        axis=1
    )  # y+h

    eps = 1e-5  # To avoid division by 0

    # area of intersection
    interArea = np.clip((xB - xA + eps), 0, 1) * np.clip((yB - yA + eps), 0, 1)

    # compute the area of both rectangles

    boxAArea = (joint_df["w"] + eps) * (joint_df["h"] + eps)  # w*h
    boxBArea = (joint_df["w_pred"] + eps) * (joint_df["h_pred"] + eps)  # w*h

    # IOU
    iou = interArea / (boxAArea + boxBArea - interArea)

    return iou


@click.command()
@click.parameter("--in_img", type=click.Path(dir_okay=False, exists=True))
@click.patameter("--out_img", type=click.Path(exists=False))
def cli(in_img, out_img):
    in_img = Path(in_img)
    assert in_img.is_file

    cfg = train.config_model(register_datasets=False)
    predictor = get_predictor(cfg=cfg)
    _, prediction_visualization = predict_file(img, predictor, cfg)

    cv2.imwrite(str(out_img), prediction_visualization)


if __name__ == "__main__":
    cli()
