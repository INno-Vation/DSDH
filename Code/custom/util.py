#!/usr/bin/env python

import torch

from detectron2.structures import Boxes


def scale_boxes(predictions, mode, factor):
    for prediction in predictions:
        if mode == 'RPN':
            boxes = prediction.proposal_boxes.tensor
        elif mode == 'ROI':
            boxes = prediction.pred_boxes.tensor

        x_0, y_0, x_1, y_1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w, h = x_1 - x_0, y_1 - y_0
        delta_x, delta_y = (w * factor) / 2, (h * factor) / 2
        x_0, y_0, x_1, y_1 = x_0 - delta_x, y_0 - delta_y, x_1 + delta_x, y_1 + delta_y

        boxes = torch.stack((x_0, y_0, x_1, y_1), dim=-1)
        boxes = Boxes(boxes)
        boxes.clip(prediction.image_size)

        if mode == 'RPN':
            prediction.proposal_boxes = boxes
        elif mode == 'ROI':
            prediction.pred_boxes = boxes
    return predictions

