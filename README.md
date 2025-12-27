# AI4Science! Unleashing Foundation Vision Models: Adaptive Transfer for Diverse Data-Limited Scientific Domains

- ## 📢 ***Currently accepted to NeurIPS 2025 with high scores (6554)!***


## Table of Contents

1. [Overview](#overview)
2. [Preparation](#preparation)
3. [Core Design](#core-design)
4. [Training](#training)
   - [Fine-tuning Methods](#fine-tuning-methods)
   - [Using Different Pre-trained Models](#using-different-pre-trained-models)
5. [Hardware & Software](#hardware--software)
6. [Citation](#citation)

---

## Overview

<img src=CLAdapter.png width="600px">

- CLAdapter refnes and adapts the rich representations learned from large-scale data to various data-limited **scientific** downstream tasks, achieving **SOTA** performance across **various fields** with **10 datasets**.
- 🚀 The proposed **CLAdapter** has helped us achieve outstanding results, including:

  - 🏆 **Champion** in the **ACM MM Challenge** 2024！
  - 🥈 **Runner-up** in the **ICCV Challenge** 2023！
  - 🥈 **Runner-up** in the **ECCV Challenge** 2023！

---

## Preparation

1. Download data.

   | Dataset              | Domains                 | Class  | Train  | Val   | Test  |
   | -------------------- | ----------------------- | ------ | ------ | ----- | ----- |
   | Tiny-ImageNet        | General                 | 200    | 100000 | 10000 | 10000 |
   | PACS                 | General OOD             | 4 × 7 | 1588   | 6355  | 2048  |
   | BreakHis             | Biomedicine             | 4      | 834    | 278   | 278   |
   | HCRF                 | Biomedicine             | 2      | 70     | 70    | 140   |
   | Apple Foliar Disease | Agricultural            | 4      | 1366   | -     | 455   |
   | WHU-RS19             | Environmental Geography | 19     | 402    | 100   | 503   |
   | KTH-TIPS-2b          | Materials Science       | 11     | 3564   | -     | 1188  |
   | InsPLAD-fault        | Industrial              | 5      | 5108   | -     | 6417  |
   | UCF101               | 3D Multimedia           | 101    | 9537   | -     | 3783  |
   | HMDB51               | 3D Multimedia           | 51     | 3570   | -     | 1530  |

2. Downloading pretrained model weights from timm library

---

## Core Design
- The implementation of 'CLAdapter' can be found in './CODE/models/modules.py'.

## Training

### Replace the fine-tuning method

(Take BreakHis as example)

vit & linear & BreakHis:

- python -m torch.distributed.launch --nproc_per_node=1 CODE/train.py --model-mode vit --finetune-mode linear --csv-dir malignant_all_5fold.csv --config-name 'config_clip_vit' --image-size 224 --epochs 100 --init-lr 1e-4 --batch-size 8 --num-workers 8 --nbatch_log 300 --warmup_epochs 2 --val_fold 0 --test_fold 1 --data-root ./ --gpu_id 0

vit & full & BreakHis:

- python -m torch.distributed.launch --nproc_per_node=1 CODE/train.py --model-mode vit --finetune-mode full --csv-dir malignant_all_5fold.csv --config-name 'config_clip_vit' --image-size 224 --epochs 100 --init-lr 1e-4 --batch-size 8 --num-workers 8 --nbatch_log 300 --warmup_epochs 2 --val_fold 0 --test_fold 1 --data-root ./ --gpu_id 0

vit & CLAdapter & BreakHis:

- python -m torch.distributed.launch --nproc_per_node=1 CODE/train.py --model-mode vit --finetune-mode cla --csv-dir malignant_all_5fold.csv --config-name 'config_clip_vit' --image-size 224 --epochs 100 --init-lr 1e-4 --batch-size 8 --num-workers 8 --nbatch_log 300 --warmup_epochs 2 --val_fold 0 --test_fold 1 --data-root ./ --gpu_id 0
- SFT (Staged Fine-tuning)
  - firt step (freeze backbone)
  - seconde step (Modify _C.MODEL.finetune = None in CODE/config_clip_vit.py and replace "None" with the "weight pth path" obtained after freeze training)

cnn & linear & BreakHis:

- python -m torch.distributed.launch --nproc_per_node=1 CODE/train.py --model-mode conv --finetune-mode linear --csv-dir malignant_all_5fold.csv --config-name 'config_clip_convnext' --image-size 224 --epochs 100 --init-lr 1e-4 --batch-size 8 --num-workers 8 --nbatch_log 300 --warmup_epochs 2 --val_fold 0 --test_fold 1 --data-root ./ --gpu_id 0

cnn & full & BreakHis:

- python -m torch.distributed.launch --nproc_per_node=1 CODE/train.py --model-mode conv --finetune-mode full --csv-dir malignant_all_5fold.csv --config-name 'config_clip_convnext' --image-size 224 --epochs 100 --init-lr 1e-4 --batch-size 8 --num-workers 8 --nbatch_log 300 --warmup_epochs 2 --val_fold 0 --test_fold 1 --data-root ./ --gpu_id 0

cnn & CLAdapter & BreakHis:

- python -m torch.distributed.launch --nproc_per_node=1 CODE/train.py --model-mode vit --finetune-mode cla --csv-dir malignant_all_5fold.csv --config-name 'config_clip_vit' --image-size 224 --epochs 100 --init-lr 1e-4 --batch-size 8 --num-workers 8 --nbatch_log 300 --warmup_epochs 2 --val_fold 0 --test_fold 1 --data-root ./ --gpu_id 0
- SFT (Staged Fine-tuning)
  - firt step (freeze backbone)
  - seconde step (Modify _C.MODEL.finetune = None in CODE/config_clip_convnext.py and replace "None" with the "weight pth path" obtained after freeze training)

### Replace the natural pre-trained model

- vit: Modify "_C.MODEL.backbone.model_name = ..." in CODE/config_clip_vit.py

  - LAION-2B
    - vit_base_patch16_clip_224.laion2b
  - LAION-400M
    - vit_base_patch16_clip_224.openai
  - ImageNet-21K
    - vit_base_patch16_224.augreg_in21k
- convnext: Modify "_C.MODEL.backbone.model_name = ..." in CODE/config_clip_convnext.py

  - LAION-2B
    - convnext_base.clip_laion2b_augreg
  - LAION-400M
    - convnext_base laion400m_s13b_b51k
  - ImageNet-21K
    - convnext_base.fb_in22k

---

## Hardware & Software

Ubuntu 20.04 LTS

GPU: 4 * 3090-24G

Python: 3.9.7

Pytorch: 1.12.1+cu116


## Citation
```

@inproceedings{li2025unleashing,
  title        = {Unleashing Foundation Vision Models: Adaptive Transfer for Diverse Data-Limited Scientific Domains},
  author       = {Li, Qiankun and He, Feng and Chen, Huabao and Ning, Xin and Wang, Kun and Wang, Zengfu},
  booktitle    = {Advances in Neural Information Processing Systems},
  year         = {2025}
}

```
