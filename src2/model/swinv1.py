import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

class SwinClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(SwinClassifier, self).__init__()
        self.model = SwinTransformer(
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

