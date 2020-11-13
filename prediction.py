from pathlib import Path
import pandas as pd
from detectron2.engine import DefaultPredictor
import cv2
from progress.bar import Bar

def get_predictor(cfg, model_name = "model_final.pth"):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    model_path = Path(cfg.OUTPUT_DIR)/model_name # path to the model we just trained
    assert model_path.is_file()

    cfg.MODEL.WEIGHTS = str(model_path) 
    
    cfg.RETINANET.SCORE_THRESH_TEST = 0.7 # used for retinanet
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold, used for maskrcnn
    predictor = DefaultPredictor(cfg)
    return predictor


def predict_submission(sub_file: Path, predictor: DefaultPredictor, img_path: Path):
    assert img_path.is_dir()
    df = pd.read_csv(str(sub_file))
    outs = []

    # TODO: this can probably be optimized passing a tensor and not
    # in a for loop ...

    bar = Bar('Processing', max=len(df))

    for base_filename in df['Image_ID']:
        file_name = img_path/f"{base_filename}.JPG"
        assert file_name.is_file()

        im = cv2.imread(str(file_name))
        outputs = predictor(im)

        pred_df = prediction_to_df(outputs)
        pred_df['Image_ID'] = base_filename

        outs.append(pred_df)
        bar.next()
    
    bar.finish()
    return pd.concat(outs, axis = 0)
        


def prediction_to_df(outputs):
    pred = outputs["instances"].to("cpu")
     # height (y), width (x)
    img_size = pred.__dict__['_image_size']
    box_scores = pred.__dict__['_fields']['scores'].numpy()
    #  Each row is (x1, y1, x2, y2).
    boxes = pred.__dict__['_fields']['pred_boxes'].__dict__['tensor'].numpy()

    out_boxes = boxes.copy()

    # Convert to relative measurements
    out_boxes[:,0::2] = out_boxes[:,0::2] / img_size[1] 
    out_boxes[:,1::2] = out_boxes[:,1::2] / img_size[0] 

    # Sub entries have to be x, y, w, h
    out_boxes[:, 2] = out_boxes[:, 2] - out_boxes[:, 0]
    out_boxes[:, 3] = out_boxes[:, 3] - out_boxes[:, 1]

    df = pd.DataFrame(out_boxes, columns=["x", "y", "w", "h"])
    df['score'] = box_scores
    return df


def predict_file(img: Path, predictor: DefaultPredictor):
    im = cv2.imread(str(img))
    outputs = predictor(im)
    return outputs

