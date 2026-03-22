import torch
import torchvision.transforms.functional as TF
import math
import timm
import torch.nn as nn
from pathlib import Path

class SquarePadResize(object):
    """
    Pads an image to a square (preserving aspect ratio) using white/black bars,
    then resizes it to the target size. This prevents "squishing" retail products.
    """
    def __init__(self, target_size, fill=255):
        self.target_size = target_size if isinstance(target_size, tuple) else (target_size, target_size)
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        img = TF.pad(img, padding, fill=self.fill, padding_mode='constant')
        return TF.resize(img, self.target_size)
        
    def __repr__(self):
        return self.__class__.__name__ + f'(target_size={self.target_size}, fill={self.fill})'

class DINOFinetuner(nn.Module):
    def __init__(self, model_name, embed_dim):
        super().__init__()
        weights_path = Path(__file__).parent / 'best_dino.pt'
        
        # INCREASED RESOLUTION: 448x448
        # This allows DINOv2 to actually "read" fine-grained text like "16 KAPSLER" vs "250G"
        self.backbone = timm.create_model(model_name, pretrained=not weights_path.exists(), num_classes=0, img_size=448)
        
        # Always output the native embedding dimension (768 for Base)
        self.head = nn.Identity()

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.head(features)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)
