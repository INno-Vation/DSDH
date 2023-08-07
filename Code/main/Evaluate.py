#!/usr/bin/env python

import os
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# path setting
dataset = 'ACNE-DET'
process = '20230118'
result = 'last'
path = ''
task_list = ['bbox']  # 'RPN', 'bbox', 'segm'

# parameter
default = False  # True, False
useCats = 1  # 0, 1
iouThrs = [0.5]  #, 0.75, 0.95
maxDets = [10, 100, 1000]
areaRng = [[0 ** 2, 1e5 ** 2], ]

gt_label_path = os.path.join(dataset, process, 'test/label.json')
coco_gt = COCO(gt_label_path)

for task in task_list:
    if result == 'last':
        result = os.listdir(path)[-1]
    result_file = 'coco_instances_results_RPN.json' if task == 'RPN' else 'coco_instances_results.json'
    dt_label_path = os.path.join(path, result, result_file)

    with open(dt_label_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    if task == 'segm':
        for d in data:
            d.pop("bbox", None)

    coco_dt = coco_gt.loadRes(data)

    iou_type = 'bbox' if task in ['RPN', 'bbox'] else 'segm'
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    if not default:
        coco_eval.params.useCats = useCats
        coco_eval.params.iouThrs = iouThrs
        coco_eval.params.maxDets = maxDets
        coco_eval.params.areaRng = areaRng

    coco_eval.evaluate()
    coco_eval.accumulate()
    # coco_eval.summarize()
    print(coco_eval.eval['recall'])
