<h1 align="left">Unofficial PyTorch Implementation of Exploring Plain Vision Transformer Backbones for Object Detection<a href="https://arxiv.org/abs/2203.16527"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a></h1> 

<p align="center">
  <a href="#Results">Results</a> |
  <a href="#Updates">Updates</a> |
  <a href='#Todo'>Todo</a> |
  <a href="#Acknowledge">Acknowledge</a>
</p>

This branch contains the **unofficial** pytorch implementation of <a href="https://arxiv.org/abs/2203.16527">Exploring Plain Vision Transformer Backbones for Object Detection</a>. Thanks for their wonderful work!

## Results from this repo on COCO

The models are trained on 4 A100 machines with 2 images per gpu, which makes a batch size of 64 during training.

| Model | Pretrain | Machine | FrameWork | Box mAP | Mask mAP | config | log | weight |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | 
| ViT-Base | IN1K+MAE | TPU | Mask RCNN | 51.1 | 45.5 | [config](./configs/ViTDet/ViTDet-ViT-Base-100e.py) | [log](logs/ViT-Base-TPU.log.json) | [OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgQuegyG-Z3FH2LDP?e=9ij98g) |
| ViT-Base | IN1K+MAE | GPU | Mask RCNN | 50.9 | 45.2 | [config](./configs/ViTDet/ViTDet-ViT-Base-100e.py) | [log](logs/ViT-Base-GPU.log.json) | [OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgQyB0R1bn1h7HzSz?e=IczYEw) |
| ViTAE-Base | IN1K+MAE | GPU | Mask RCNN | 51.2 | 45.6 | [config](configs/ViTDet/ViTDet-ViTAE-Base-100e.py) | [log](logs/ViTAE-Base-GPU.log.json) | [OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgQ0mNFX8iGvMAGmI) |

## Updates

> [2022-04-17] Release the pretrained weights and logs for ViT-B and ViTAE-B on MS COCO. The models are totally trained with PyTorch on GPU.

> [2022-04-16] Release the initial unofficial implementation of ViTDet with ViT-Base model! It obtains 51.1 mAP and 45.5 mAP on detection and segmentation, respectively. The weights and logs will be uploaded soon. 

> Applications of ViTAE Transformer include: [image classification](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Image-Classification) | [object detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Object-Detection) | [semantic segmentation](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Semantic-Segmentation) | [animal pose segmentation](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Animal-Pose-Estimation) | [remote sensing](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing) | [matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting)

## Todo

This repo current contains modifications including:

- using LN for the convolutions in RPN and heads
- using large scale jittor for augmentation
- using RPE from MViT
- using longer training epochs and 1024 test size
- using global attention layers

There are other things to do:

- [ ] Implement the conv blocks for global information communication

- [ ] Tune the models for Cascade RCNN 

- [ ] Train ViT models for the LVIS dataset

- [ ] Train ViTAE model with the ViTDet framework

## Acknowledge
We acknowledge the excellent implementation from [mmdetection](https://github.com/open-mmlab/mmdetection), [MAE](https://github.com/facebookresearch/mae), [MViT](https://github.com/facebookresearch/mvit), and [BeiT](https://github.com/microsoft/unilm/tree/master/beit).

## Citing ViTDet
```
@article{Li2022ExploringPV,
  title={Exploring Plain Vision Transformer Backbones for Object Detection},
  author={Yanghao Li and Hanzi Mao and Ross B. Girshick and Kaiming He},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.16527}
}
```