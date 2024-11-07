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
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Separate linear projections for query, key, and value
        self.linear_q = nn.Linear(embed_dim, embed_dim)  # Query projection
        self.linear_k = nn.Linear(embed_dim, embed_dim)  # Key projection
        self.linear_v = nn.Linear(embed_dim, embed_dim)  # Value projection

    def forward(self, x):
        """
        Args:
            x: Input feature map of shape (batch_size, channels, height, width)
        Returns:
            Output tensor of shape (batch_size, channels, height, width)
        """
        # Reshape x to a sequence format for multi-head attention
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(2, 0, 1)  # Shape: (h * w, b, c)

        # Project input for query, key, and value
        q = self.linear_q(x_flat)
        k = self.linear_k(x_flat)
        v = self.linear_v(x_flat)

        # Apply multi-head self-attention
        attn_output, _ = self.multihead_attn(q, k, v)  # Self-attention since q, k, v are derived from x

        # Reshape the output back to the original image shape
        attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)  # Shape: (b, c, h, w)

        # Residual connection
        return x + attn_output  # Adding self-attention output to the original input


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
    
    
class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResidualBlock, n=1):
        super(ResNetLayer, self).__init__()
        downsampling = 2 if in_channels != out_channels else 1
        layers = [block(in_channels, out_channels)]
        for _ in range(1, n):
            layers.append(block(out_channels, out_channels))
        self.blocks = nn.Sequential(*layers)
        self.downsample = nn.MaxPool2d(kernel_size=downsampling) if downsampling > 1 else nn.Identity()

    def forward(self, x):
        x = self.blocks(x)
        return self.downsample(x)


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels=3, num_heads=1):
        super(FeatureEncoder, self).__init__()

        # DoubleConv layers
        self.dconv_down_1 = DoubleConv(3, 64)
        self.dconv_down_2 = DoubleConv(64, 128)
        self.dconv_down_3 = DoubleConv(128, 256)
        self.dconv_down_4 = DoubleConv(256, 512)

        # ResNet layers with ResNetLayer
        self.resnet_layer1 = ResNetLayer(64, 64, n=2)
        self.resnet_layer2 = ResNetLayer(128, 128, n=2)
        self.resnet_layer3 = ResNetLayer(256, 256, n=2)
        self.resnet_layer4 = ResNetLayer(512, 512, n=2)

        # Self-attention layers
        self.self_attn_1 = SelfAttention(64, num_heads)
        self.self_attn_2 = SelfAttention(128, num_heads)
        self.self_attn_3 = SelfAttention(256, num_heads)
        self.self_attn_4 = SelfAttention(512, num_heads)

        # Pooling layer for selective downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoding stage 1
        x = self.dconv_down_1(x)
        x = self.pool(x)
        x = self.resnet_layer1(x)
        c4 = self.self_attn_1(x)

        # Encoding stage 2
        x = self.dconv_down_2(c4)
        x = self.pool(x)
        x = self.resnet_layer2(x)
        c3 = self.self_attn_2(x)
        
        # Encoding stage 3
        x = self.dconv_down_3(c3)
        x = self.pool(x)
        x = self.resnet_layer3(x)
        c2 = self.self_attn_3(x)
        
        # Encoding stage 4
        x = self.dconv_down_4(c2)
        x = self.resnet_layer4(x)
        c1 = self.self_attn_4(x)

        return c1, c2, c3, c4







 





