import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(2, 0, 1)
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = attn_output.permute(1, 2, 0).reshape(b, c, h, w)
        return x.permute(1, 2, 0).reshape(b, c, h, w) + attn_output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class FeatureEncoder(nn.Module):
    def __init__(self, in_channels=3, num_heads=4):
        super(FeatureEncoder, self).__init__()

        # DoubleConv layers followed by pooling for each encoding stage
        self.dconv_down_1 = DoubleConv(in_channels, 64)
        self.dconv_down_2 = DoubleConv(64, 128)
        self.dconv_down_3 = DoubleConv(128, 256)
        self.dconv_down_4 = DoubleConv(256, 512)

        # ResNet layers to keep spatial dimensions consistent
        self.res1 = ResidualBlock(64, 64)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(256, 256)

        # Self-attention layers for each encoding stage
        self.self_attn_1 = SelfAttention(64, num_heads)
        self.self_attn_2 = SelfAttention(128, num_heads)
        self.self_attn_3 = SelfAttention(256, num_heads)

        # Pooling layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoding stage 1
        x = self.dconv_down_1(x)
        x = self.res1(x)
        x = self.self_attn_1(x)  # Self-attention without changing channels
        c1 = self.pool(x)  # Shape should be [batch_size, 64, 32, 32]

        # Encoding stage 2
        x = self.dconv_down_2(c1)
        x = self.res2(x)
        x = self.self_attn_2(x)  # Self-attention without changing channels
        c2 = self.pool(x)  # Shape should be [batch_size, 128, 16, 16]

        # Encoding stage 3
        x = self.dconv_down_3(c2)
        x = self.res3(x)
        x = self.self_attn_3(x)  # Self-attention without changing channels
        c3 = self.pool(x)  # Shape should be [batch_size, 256, 8, 8]

        # Encoding stage 4 (no additional pooling)
        c4 = self.dconv_down_4(c3)  # Shape should be [batch_size, 512, 8, 8]

        return c1, c2, c3, c4




