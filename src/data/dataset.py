import pandas as pd
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .transforms import *


class Nyu2Dataset(Dataset):

    def __init__(self, csv_file: str, transform=None):
        self.paths = pd.read_csv(csv_file, header=None,
                                 names=['image', 'depth'])
        self.transform = transform

    def __getitem__(self, idx: int) -> dict:

        image = Image.open(self.paths['image'][idx])
        depth = Image.open(self.paths['depth'][idx])
        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.paths)


class Nyu2DataModule(LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def train_dataloader(self):
        config = self.config

        train_transform = transforms.Compose(
            [
                Scale(config.DATA.SCALE_SIZE_MIN),
                RandomHorizontalFlip(),
                RandomRotate(config.DATA.RANDOM_ROT_DEGREE),
                CenterCrop(
                    config.DATA.CENTER_CROP_SIZE,
                    config.DATA.OUTPUT_SIZE),
                ToTensor(),
                Lighting(config.DATA.PCA_LIGHTING,
                         torch.Tensor(config.DATA.PCA_EIGVAL),
                         torch.Tensor(config.DATA.PCA_EIGVEC)),
                ColorJitter(
                    brightness=config.DATA.RANDOM_BRIGHTNESS,
                    contrast=config.DATA.RANDOM_CONTRAST,
                    saturation=config.DATA.RANDOM_SATURATION,
                ),
                Normalize(config.DATA.NORMALIZE_MEAN,
                          config.DATA.NORMALIZE_STD)

            ]
        )
        train_dataset = Nyu2Dataset(
            csv_file=config.DATASET.TRAIN_CSV,
            transform=train_transform)
        train_loader = DataLoader(
            train_dataset,
            config.SOLVER.BATCHSIZE,
            shuffle=True,
            num_workers=config.SOLVER.NUM_WORKERS,
            pin_memory=False)

        return train_loader

    def val_dataloader(self):
        config = self.config

        test_transform = transforms.Compose(
            [
                Scale(config.DATA.SCALE_SIZE_MIN),
                CenterCrop(
                    config.DATA.CENTER_CROP_SIZE,
                    config.DATA.CENTER_CROP_SIZE),
                ToTensor(is_test=True),
                Normalize(config.DATA.NORMALIZE_MEAN,
                          config.DATA.NORMALIZE_STD)
            ]
        )
        test_dataset = Nyu2Dataset(
            csv_file=config.DATASET.TEST_CSV,
            transform=test_transform)
        test_loader = DataLoader(
            test_dataset,
            config.TEST.BATCHSIZE,
            shuffle=False,
            num_workers=config.SOLVER.NUM_WORKERS,
            pin_memory=False)

        return test_loader
