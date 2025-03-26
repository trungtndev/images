import torch
from torch import nn
from torch.nn import functional as F
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_rate=16):
        super(ChannelAttention, self).__init__()
        # self.squeeze = nn.ModuleList([
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.AdaptiveMaxPool2d(1)
        # ])
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=channels // reduction_rate,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels // reduction_rate,
                      out_channels=channels,
                      kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # perform squeeze with independent Pooling
        avg_feat = F.avg_pool2d(x, x.size()[2:], ceil_mode=True)
        max_feat = F.max_pool2d(x, x.size()[2:], ceil_mode=True)
        # perform excitation with the same excitation sub-net
        avg_out = self.excitation(avg_feat)
        max_out = self.excitation(max_feat)
        # attention
        attention = self.sigmoid(avg_out + max_out)
        return attention * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.bn1 = nn.BatchNorm2d(2)
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_feat    = torch.mean(x, dim=1, keepdim=True)
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_feat, max_feat], dim=1)
        feat = self.bn1(feat)
        feat = self.conv(feat)
        feat = self.bn2(feat)
        attention = self.sigmoid(feat)
        return attention * x

class CBAM(nn.Module):
    def __init__(self, channels, reduction_rate=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels,
                                                  reduction_rate)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)

        return out