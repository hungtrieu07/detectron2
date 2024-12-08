# Some basic setup:
# Setup detectron2 logger
import numpy as np
# import cv2

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

import random

setup_logger()

# download dataset (remove comment if you want to download dataset from roboflow)
# from roboflow import Roboflow

# rf = Roboflow(api_key="QubW6TSf6XudrmCK6KmK")
# project = rf.workspace("firesmokehuman").project("humantraffic")
# version = project.version(12)
# dataset = version.download("coco")

# register dataset
from detectron2.data.datasets import register_coco_instances

register_coco_instances(
    "my_dataset_train",
    {},
    "HumanTraffic-19/train/_annotations.coco.json",
    "HumanTraffic-19/train",
)
register_coco_instances(
    "my_dataset_val",
    {},
    "HumanTraffic-19/valid/_annotations.coco.json",
    "HumanTraffic-19/valid",
)
register_coco_instances(
    "my_dataset_test",
    {},
    "HumanTraffic-19/test/_annotations.coco.json",
    "HumanTraffic-19/test",
)

# We are importing our own Trainer Module here to use the COCO validation evaluation during training.
# Otherwise no validation eval occurs.

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


# TRAIN MODEL 
# from .detectron2.tools.train_net import Trainer
# from detectron2.engine import DefaultTrainer

from detectron2.config import get_cfg

# from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os

cfg = get_cfg()
# select from modelzoo here: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#coco-object-detection-baselines
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
)
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)

cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001

cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = (
    10000  # adjust up if val mAP is still rising, adjust down if overfit
)
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = (
    6  # your number of classes (Number of foreground classes)
)

cfg.TEST.EVAL_PERIOD = 500

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# test evaluation
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_test")
inference_on_dataset(trainer.model, val_loader, evaluator)

# # inference
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.DATASETS.TEST = ("my_dataset_test",)
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
# predictor = DefaultPredictor(cfg)
# test_metadata = MetadataCatalog.get("my_dataset_test")

# from detectron2.utils.visualizer import ColorMode

# import glob

# for imageName in glob.glob("/content/test/*jpg"):
#     im = cv2.imread(imageName)
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1], metadata=test_metadata, scale=0.9)
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2_imshow(out.get_image()[:, :, ::-1])
