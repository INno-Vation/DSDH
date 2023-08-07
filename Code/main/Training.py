#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import os
import random
import numpy as np
from collections import OrderedDict
import torch
import platform
import datetime

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.data.datasets import register_coco_instances


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            now = datetime.datetime.now()
            output_folder = now.strftime("%Y-%m-%d_%H-%M-%S")
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", output_folder)
            os.makedirs(output_folder)

        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        print('evaluator_type:', evaluator_type)
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def register_dataset():
    dataset_dir = os.path.join(root_dir, '../../dataset', dataset, process)
    train_label_path = os.path.join(dataset_dir, train_on, 'label.json')
    train_image_dir = os.path.join(dataset_dir, train_on, 'image')
    test_label_path = os.path.join(dataset_dir, test_on, 'label.json')
    test_image_dir = os.path.join(dataset_dir, test_on, 'image')

    register_coco_instances('dataset_train', {}, train_label_path, train_image_dir)
    register_coco_instances('dataset_test', {}, test_label_path, test_image_dir)

    DatasetCatalog.get("dataset_train")
    MetadataCatalog.get("dataset_train")
    DatasetCatalog.get("dataset_test")
    MetadataCatalog.get("dataset_test")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    cfg.DATASETS.TRAIN = ('dataset_train',)
    cfg.DATASETS.TEST = ('dataset_test',)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 3
    cfg.MODEL.BACKBONE.FREEZE_AT = 1
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    train_num = 5020
    epoch_step = int(train_num / cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.CHECKPOINT_PERIOD = int(epoch_step / 2)
    cfg.TEST.EVAL_PERIOD = int(epoch_step / 2)
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.ROI_HEADS.NAME = "My_StandardROIHeads"
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = [0.01,]
    cfg.MODEL.ROI_BOX_HEAD.NAME_reg = "FastRCNNConvFCHead_reg"
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV_reg = 0
    cfg.MODEL.ROI_BOX_HEAD.CONV_DIM_reg = 256
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC_reg = 2
    cfg.MODEL.ROI_BOX_HEAD.FC_DIM_reg = 1024
    cfg.MODEL.ROI_BOX_HEAD.NORM_reg = ""
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (8 * epoch_step, 11 * epoch_step)
    cfg.SOLVER.MAX_ITER = (13 * epoch_step)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (256)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

    cfg.merge_from_list(args.opts)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    register_dataset()

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":

    # path setting
    root_dir = os.path.abspath('..')
    dataset = 'ACNE-DET'
    process = '20230118'
    train_on = 'train'
    test_on = 'test'

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

