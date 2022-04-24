<h1 align="left">Unofficial PyTorch Implementation of Exploring Plain Vision Transformer Backbones for Object Detection<a href="https://arxiv.org/abs/2203.16527"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a></h1> 

<p align="center">
  <a href="#Results">Results</a> |
  <a href="#Updates">Updates</a> |
  <a href="#Usage">Usage</a> |
  <a href='#Todo'>Todo</a> |
  <a href="#Acknowledge">Acknowledge</a>
</p>

This branch contains the **unofficial** pytorch implementation of <a href="https://arxiv.org/abs/2203.16527">Exploring Plain Vision Transformer Backbones for Object Detection</a>. Thanks for their wonderful work!

## Results from this repo on COCO

The models are trained on 4 A100 machines with 2 images per gpu, which makes a batch size of 64 during training.

| Model | Pretrain | Machine | FrameWork | Box mAP | Mask mAP | config | log | weight |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | 
| ViT-Base | IN1K+MAE | TPU | Mask RCNN | 51.1 | 45.5 | [config](./configs/ViTDet/ViTDet-ViT-Base-100e.py) | [log](logs/ViT-Base-TPU.log.json) | [OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgQuegyG-Z3FH2LDP?e=9ij98g) |
| ViT-Base | IN1K+MAE | GPU | Mask RCNN | 51.1 | 45.4 | [config](./configs/ViTDet/ViTDet-ViT-Base-100e.py) | [log](logs/ViT-Base-GPU.log.json) | [OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgRA7Y9s2rA5NC4wn?e=QfpKJf) |
| [ViTAE-Base](https://arxiv.org/abs/2202.10108) | IN1K+MAE | GPU | Mask RCNN | 51.6 | 45.8 | [config](configs/ViTDet/ViTDet-ViTAE-Base-100e.py) | [log](logs/ViTAE-Base-GPU.log.json) | [OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgQ--Ez4mzEnO-G5Y?e=ACfLxC) |
| [ViTAE-Small](https://arxiv.org/abs/2202.10108) | IN1K+Sup | GPU | Mask RCNN | 45.6 | 40.1 | [config](configs/ViTDet/ViTDet-ViTAE-Small-100e.py) | [log](logs/ViTAE-S-GPU.log.json) | [OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgQ7PorGY53K6gIGd?e=lw81U5) |

## Updates

> [2022-04-18] Explore using small 1K supervised trained models (20M parameters) for ViTDet (**45.6 mAP**). The results with multi-stage structure is **46.0 mAP** for [Swin-T](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) and **47.8 mAP** for [ViTAEv2-S](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Object-Detection) with Mask RCNN on COCO.

> [2022-04-17] Release the pretrained weights and logs for ViT-B and ViTAE-B on MS COCO. The models are totally trained with PyTorch on GPU.

> [2022-04-16] Release the initial unofficial implementation of ViTDet with ViT-Base model! It obtains 51.1 mAP and 45.5 mAP on detection and segmentation, respectively. The weights and logs will be uploaded soon. 

> Applications of ViTAE Transformer include: [image classification](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Image-Classification) | [object detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Object-Detection) | [semantic segmentation](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Semantic-Segmentation) | [animal pose segmentation](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Animal-Pose-Estimation) | [remote sensing](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing) | [matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting)

## Usage

We use PyTorch 1.9.0 or NGC docker 21.06, and mmcv 1.3.9 for the experiments.
```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.3.9
MMCV_WITH_OPS=1 pip install -e .
cd ..
git clone https://github.com/ViTAE-Transformer/ViTDet.git
cd ViTDet
pip install -v -e .
```

After install the two repos, install timm and einops, i.e.,
```bash
pip install timm==0.4.9 einops
```

Download the pretrained models from [MAE](https://github.com/facebookresearch/mae) or [ViTAE](https://github.com/ViTAE-Transformer/ViTAE-Transformer), and then conduct the experiments by

```bash
# for single machine
bash tools/dist_train.sh <Config PATH> <NUM GPUs> --cfg-options model.pretrained=<Pretrained PATH>

# for multiple machines
python -m torch.distributed.launch --nnodes <Num Machines> --node_rank <Rank of Machine> --nproc_per_node <GPUs Per Machine> --master_addr <Master Addr> --master_port <Master Port> tools/train.py <Config PATH> --cfg-options model.pretrained=<Pretrained PATH> --launcher pytorch
```

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

For ViTAE and ViTAEv2, please refer to:
```
@article{xu2021vitae,
  title={Vitae: Vision transformer advanced by exploring intrinsic inductive bias},
  author={Xu, Yufei and Zhang, Qiming and Zhang, Jing and Tao, Dacheng},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

@article{zhang2022vitaev2,
  title={ViTAEv2: Vision Transformer Advanced by Exploring Inductive Bias for Image Recognition and Beyond},
  author={Zhang, Qiming and Xu, Yufei and Zhang, Jing and Tao, Dacheng},
  journal={arXiv preprint arXiv:2202.10108},
  year={2022}
}
```
