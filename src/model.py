import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpProjWithSkip(nn.Module):
    """
    implement up projection -like module with skip connection
    up-projection module: https://arxiv.org/abs/1606.00373
    """
    def __init__(self, in_and_skip_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_and_skip_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_and_skip_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = F.relu(x1 + x2)
        return x


class MyUnet(smp.Unet):
    """
    Customize Unet by replacing  normal decode blocks with the above UpProjWithSkip
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for i, block in enumerate(self.decoder.blocks):
            self.decoder.blocks[i] = UpProjWithSkip(block.conv1[0].in_channels,
                                                    block.conv1[0].out_channels)
