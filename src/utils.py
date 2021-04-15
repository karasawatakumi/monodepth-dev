from typing import Optional, List

from omegaconf import OmegaConf, DictConfig


def load_config(cfg_path: Optional[str] = None,
                default_cfg_path: str = 'configs/default.yaml',
                update_dotlist: Optional[List[str]] = None) -> DictConfig:

    config = OmegaConf.load(default_cfg_path)
    if cfg_path is not None:
        optional_config = OmegaConf.load(cfg_path)
        config = OmegaConf.merge(config, optional_config)
    if update_dotlist is not None:
        update_config = OmegaConf.from_dotlist(update_dotlist)
        config = OmegaConf.merge(config, update_config)

    OmegaConf.set_readonly(config, True)

    return config


def print_config(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

