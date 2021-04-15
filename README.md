# Monocular Depth Estimation for [NYU2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
This repository provides a simple PyTorch Lightning implementation of monocular depth estimation for NYU Depth Dataset V2.

## Dependencies

- Docker 20.10.2
- [Docker Compose](https://docs.docker.com/compose/) 1.28.3
- Python 3.8.0
- [Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch) 0.1.3 
- [PyTorch](https://pytorch.org/) 1.6.0
- [PyTorch Lightning](https://www.pytorchlightning.ai/) 1.2.5
- [OmegaConf](https://omegaconf.readthedocs.io/en/2.0_branch/) 2.0.6
- [Weights & Biases](https://wandb.ai/site) 0.10.25

Please see [requirements.txt](./docker/requirements.txt) for the other libraries' versions.

## Approach

Segmentation model + *Depth loss

*We use the three loss proposed in [[J. Hu+, 2019]](https://arxiv.org/abs/1803.08673).

BACKBONE | TYPE | delta1 | delta2 | delta3 | lg10 | abs_rel | mae | mse
-- | -- | -- | -- | -- | -- | -- | -- | --
efficientnet-b7 | UnetPlusPlus | 0.8381 | 0.9658 | 0.9914 | 0.0553 | 0.1295 | 0.3464 | 0.3307
efficientnet-b7 | FPN | 0.8378 | 0.9662 | 0.9915 | 0.0561 | 0.1308 | 0.3523 | 0.3308
efficientnet-b4 | UnetPlusPlus | 0.8361 | 0.9649 | 0.9913 | 0.0559 | 0.1308 | 0.3488 | 0.3293
efficientnet-b4 | Unet | 0.8312 | 0.9636 | 0.9905 | 0.0569 | 0.1321 | 0.3582 | 0.3508
efficientnet-b4 | FPN | 0.8308 | 0.9648 | 0.9909 | 0.0570 | 0.1337 | 0.3581 | 0.3411
efficientnet-b4 | DeepLabV3Plus | 0.8304 | 0.9634 | 0.9900 | 0.0570 | 0.1352 | 0.3596 | 0.3483
resnet50 | FPN | 0.8287 | 0.9637 | 0.9905 | 0.0577 | 0.1351 | 0.3600 | 0.3456
resnet50 | Unet | 0.8277 | 0.9619 | 0.9903 | 0.0576 | 0.1345 | 0.3570 | 0.3421
resnet50 | MyUnet | 0.8273 | 0.9612 | 0.9894 | 0.0577 | 0.1343 | 0.3576 | 0.3458
resnet50 | UnetPlusPlus | 0.8241 | 0.9623 | 0.9896 | 0.0581 | 0.1356 | 0.3610 | 0.3486
resnet50 | DeepLabV3Plus | 0.8225 | 0.9608 | 0.9888 | 0.0583 | 0.1375 | 0.3639 | 0.3569
efficientnet-b0 | UnetPlusPlus | 0.8190 | 0.9607 | 0.9894 | 0.0592 | 0.1396 | 0.3722 | 0.3667
efficientnet-b0 | FPN | 0.8132 | 0.9597 | 0.9897 | 0.0601 | 0.1415 | 0.3780 | 0.3728

NOTE: To simplify the experiment, we set the image size to [288, 224] (divisible by 32), which is not exactly the same as the evaluation in the paper.

## Preparation

### Dataset: NYU Depth Dataset V2

```bash
sh scripts/prepare_nyu2.sh
```

[This script](./scripts/prepare_nyu2.sh) uses the downloading link in [J. Hu's repository](https://github.com/JunjH/Revisiting_Single_Depth_Estimation).


### Installation

```bash
docker-compose build
docker-compose run dev
```

## Run

### Train
```bash
python tools/train.py
```

```bash
usage: train.py [-h] [--config CONFIG] [--resume RESUME] [--gpu-ids GPU_IDS [GPU_IDS ...] | --n-gpu N_GPU] [--amp {O1,O2,O3}]
                [--profiler {simple,advanced}]
                ...

Train a predictor

positional arguments:
  opts                  Overwrite configs. (ex. OUTPUT_DIR=results, SOLVER.NUM_WORKERS=8)

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Optional config path. `configs/default.yaml` is loaded by default
  --resume RESUME       the checkpoint file to resume from
  --gpu-ids GPU_IDS [GPU_IDS ...]
  --n-gpu N_GPU
  --amp {O1,O2,O3}      amp opt level
  --profiler {simple,advanced}
                        'simple' or 'advanced'
```

If you want to override the config with command line args, put them at the end in the form of dotlist.

```bash
python tools/train.py --config [config path] SOLVER.NUM_WORKERS=8 SOLVER.EPOCH=5
```


## Credit

```
@inproceedings{Hu2019RevisitingSI,
  title={Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps With Accurate Object Boundaries},
  author={Junjie Hu and Mete Ozay and Yan Zhang and Takayuki Okatani},
  journal={2019 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2019}
}
```
