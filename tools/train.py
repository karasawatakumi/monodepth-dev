import argparse
import os
import torch
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data import Nyu2DataModule
from src.plmodel import DepthPLModel
from src.utils import load_config, print_config

WANDB_PJ_NAME = 'monodepth-dev'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a predictor')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional config path. `configs/default.yaml` is loaded by default.')
    parser.add_argument('--resume', type=str, default=None, help='the checkpoint file to resume from')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu-ids', type=int, default=None, nargs='+')
    group.add_argument('--n-gpu', type=int, default=None)
    parser.add_argument("--amp", default=None, help="amp opt level", choices=['O1', 'O2', 'O3'])
    parser.add_argument("--profiler", default=None, help="'simple' or 'advanced'", choices=['simple', 'advanced'])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Overwrite configs. (ex. OUTPUT_DIR=results, SOLVER.NUM_WORKERS=8)')
    return parser.parse_args()


def get_gpus(args):
    if args.gpu_ids is not None:
        gpus = args.gpu_ids
    elif args.n_gpu is not None:
        gpus = args.n_gpu
    else:
        gpus = 1
    gpus = gpus if torch.cuda.is_available() else None
    return gpus


def get_trainer(args, config) -> Trainer:

    # amp
    precision = 16 if args.amp is not None else 32

    # logger
    if not args.debug:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        w_logger = WandbLogger(project=WANDB_PJ_NAME, save_dir=config.OUTPUT_DIR, name=config.OUTPUT_DIR)
        w_logger.log_hyperparams(OmegaConf.to_container(config))
    else:
        w_logger = False

    # checkpoint
    ckpt_callback = ModelCheckpoint(filename='{epoch:03d}-{rmse:.3f}-{delta1:.3f}',
                                    save_top_k=1, monitor='delta1', mode='max')

    return Trainer(
        max_epochs=config.SOLVER.EPOCH,
        callbacks=ckpt_callback,
        resume_from_checkpoint=args.resume,
        default_root_dir=config.OUTPUT_DIR,
        gpus=get_gpus(args),
        precision=precision,
        amp_level=args.amp,
        profiler=args.profiler,
        logger=w_logger,
        fast_dev_run=args.debug,
    )


def main():
    args = parse_args()

    # config
    config: DictConfig = load_config(args.config, update_dotlist=args.opts)
    print_config(config)

    # modules
    model = DepthPLModel(config)
    nyu2data = Nyu2DataModule(config)

    # trainer setting
    trainer = get_trainer(args, config)

    # train
    trainer.fit(model, nyu2data)


if __name__ == "__main__":
    main()
