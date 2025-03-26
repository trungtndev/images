import torch.nn as nn
from timm import create_model

class SwinClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(SwinClassifier, self).__init__()
        self.model = create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

