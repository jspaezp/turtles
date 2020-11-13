from pathlib import Path

from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.engine import DefaultTrainer
import click


def config_model(
    modelzoo_file="COCO-Detection/retinanet_R_101_FPN_3x.yaml",
    train_dataset="turtles_train",
    test_dataset="turtles_test",
):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(modelzoo_file))
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (test_dataset,)

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        modelzoo_file
    )  # Let training initialize from model zoo

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset

    """
    # Section specific for FASTER-RCNN
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128  # faster, and good enough for this toy dataset (default: 512)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    """

    # Section for retinanet
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.IOU_THRESHOLDS = [0.4, 0.5]
    # Select topk candidates before NMS
    cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 500  # Default 1000
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.2  # Default 0.5
    # Options are: "smooth_l1", "giou"
    cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE = "giou"  # Default smooth_l1

    cfg.INPUT.CROP.SIZE = [0.7, 1]
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False # Defaults to False

    Path(cfg.OUTPUT_DIR).mkdir(exist_ok=True)
    return cfg


def make_trainer(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer


@click.command
@click.option("--split_dir", type=click.Path(file_okay=False, exists=True))
@click.option("--out_dir", type=click.Path(file_okay=False, exists=True))
def cli(split_dir, out_dir):
    register_datasets(Path(split_dir))
    cfg = config_model()
    trainer = make_trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    from t_io import register_datasets