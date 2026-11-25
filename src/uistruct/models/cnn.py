"""Model 3 type head: the Model 1 MLP plus a pretrained CNN embedding of each element's crop.

Backbone: torchvision ResNet-18 with ImageNet weights only (no UI data, so nothing about CLAY/RICO leaks in),
classification layer removed -> 512-d pooled feature. Crops arrive as uint8 (B, S, S, 3) and are normalised
on the GPU with ImageNet statistics (nothing fitted on our data). The 512-d feature goes through a BatchNorm
(running statistics from training batches only) and is concatenated with the numeric inputs of the MLP.

Two uses, same head:
  frozen      the backbone is run once (`embed`) and the MLP trains on stored features (`img` = features)
  fine-tuned  backbone and MLP train together (`img` = crops), backbone at a smaller learning rate
"""
import numpy as np
import torch
from torch import nn

from uistruct.data.crops import IMAGENET_MEAN, IMAGENET_STD
from uistruct.models.type_mlp import TypeMLP

FEAT_DIM = 512


def resnet18_trunk(pretrained=True):
    import torchvision
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=w)
    m.fc = nn.Identity()
    return m


class Normalise(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255)

    def forward(self, x):            # x: uint8 (B, S, S, 3)
        return (x.permute(0, 3, 1, 2).float() - self.mean) / self.std


class CropTypeNet(nn.Module):
    """backbone=None: `img` is a stored (B, 512) feature (frozen use). Otherwise `img` is a uint8 crop batch."""

    def __init__(self, n_num, n_names, n_pieces, backbone=None, **mlp_kw):
        super().__init__()
        self.norm = Normalise()
        self.backbone = backbone
        self.feat_bn = nn.BatchNorm1d(FEAT_DIM)
        self.mlp = TypeMLP(n_num + FEAT_DIM, n_names, n_pieces, **mlp_kw)

    def forward(self, num, name, pieces, img):
        f = self.backbone(self.norm(img)) if self.backbone is not None else img.float()
        return self.mlp(torch.cat([num, self.feat_bn(f.float())], 1), name, pieces)


@torch.no_grad()
def embed(backbone, crops, device, batch=1024, amp=True):
    """Frozen features for a uint8 crop array (n, S, S, 3) -> float16 (n, 512)."""
    norm = Normalise().to(device)
    backbone = backbone.to(device).eval()
    out = []
    for i in range(0, len(crops), batch):
        x = torch.from_numpy(np.array(crops[i:i + batch])).to(device, non_blocking=True)   # copy: memmap is read-only
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out.append(backbone(norm(x)).half().cpu())
    return torch.cat(out).numpy()
