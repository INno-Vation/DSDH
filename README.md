# Decoupled Sequential Detection Head for Accurate Acne Detection

This project contains a Pytorch implementation of the paper:
> Decoupled Sequential Detection Head for Accurate Acne Detection

## Introduction
This code is implemented based on Detectron2:
> https://github.com/facebookresearch/detectron2

## DSDH
Our Decoupled Sequential Detection Head (DSDH) is located in:
> Code/detectron2/modeling/roi_heads/roi_heads.py/My_StandardROIHeads

## Training
```shell
python Code/main/Training.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
```

## Evaluate
```shell
python Code/main/Evaluate.py
```

## Dataset
Our dataset ACNE-DET (labelme format) could be accessed at [Baidu](https://pan.baidu.com/s/19fv9itHcpjAQCbBxIzvb2Q) (password: foa7).