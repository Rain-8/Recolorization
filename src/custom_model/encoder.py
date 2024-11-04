import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => Instance Norm => Leaky ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class CrossAttention(nn.Module):
    """Applies cross-attention between a feature map and palette embedding."""
    def __init__(self, embed_dim, palette_embed, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Query, Key, Value projections for attention mechanism
        self.linear_q = nn.Linear(palette_embed, embed_dim)  # Query projection for palette embedding
        self.linear_k = nn.Linear(embed_dim, embed_dim)  # Key projection for feature map
        self.linear_v = nn.Linear(embed_dim, embed_dim)  # Value projection for feature map

    def forward(self, x, palette_embedding):
        b, c, h, w = x.shape
        # Reshape and project feature map
        x_flat = x.view(b, c, h * w).permute(2, 0, 1)
        k = self.linear_k(x_flat)  # Key projection
        v = self.linear_v(x_flat)  # Value projection

        # Reshape and project palette embedding
        b_, c_, h_, w_ = palette_embedding.shape
        palette_embedding = palette_embedding.view(b_, c_, h_ * w_).permute(2, 0, 1)
        q = self.linear_q(palette_embedding)  # Query projection

        # Multi-head attention
        attn_output, _ = self.multihead_attn(q, k, v)
        attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)

        # Concatenate attention output with the original feature map
        return torch.cat([x, attn_output], dim=1)

def adjust_target_palettes(target_palettes_emb, h, w):
    """
    Expands target palette embedding to match the spatial dimensions.
    
    Args:
        target_palettes_emb (torch.Tensor): The target palette embedding tensor of shape (batch, channels).
        h (int): The height to which the embedding should be expanded.
        w (int): The width to which the embedding should be expanded.
    
    Returns:
        torch.Tensor: Expanded target palette embedding tensor of shape (batch, channels, h, w).
    """
    return target_palettes_emb.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)

class RecoloringEncoder(nn.Module):
    def __init__(self, in_channels=3, palette_embedding_dim=64, num_heads=4):
        super().__init__()
        self.palette_embedding_dim = palette_embedding_dim
        self.palette_fc = nn.Linear(4 * 60 * 4, palette_embedding_dim)

        # DoubleConv layers for each encoding stage
        self.dconv_down_1 = DoubleConv(in_channels, 64)
        self.dconv_down_2 = DoubleConv(128, 128)
        self.dconv_down_3 = DoubleConv(192, 256)
        self.dconv_down_4 = DoubleConv(320, 256)

        # Cross-attention layers to condition with palette information
        self.cross_attn_1 = CrossAttention(64, palette_embedding_dim, num_heads)
        self.cross_attn_2 = CrossAttention(128, palette_embedding_dim, num_heads)
        self.cross_attn_3 = CrossAttention(256, palette_embedding_dim, num_heads)
        self.cross_attn_4 = CrossAttention(256, palette_embedding_dim, num_heads)

        # Pooling layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, target_palettes):
        bz, _, _, _ = x.shape

        # Flatten and project target palettes to create a conditioning embedding
        target_palettes_flat = target_palettes.view(bz, -1)
        palette_embedding = self.palette_fc(target_palettes_flat)

        # Encoding stages with cross-attention
        x = self.dconv_down_1(x)
        palette_embedding_repeated = adjust_target_palettes(palette_embedding, x.size(-2), x.size(-1))
        x = self.cross_attn_1(x, palette_embedding_repeated)
        c1 = self.pool(x)  # c1 is saved for use in the decoder

        x = self.dconv_down_2(c1)
        palette_embedding_repeated = adjust_target_palettes(palette_embedding, x.size(-2), x.size(-1))
        x = self.cross_attn_2(x, palette_embedding_repeated)
        c2 = self.pool(x)

        x = self.dconv_down_3(c2)
        palette_embedding_repeated = adjust_target_palettes(palette_embedding, x.size(-2), x.size(-1))
        x = self.cross_attn_3(x, palette_embedding_repeated)
        c3 = self.pool(x)

        x = self.dconv_down_4(c3)
        palette_embedding_repeated = adjust_target_palettes(palette_embedding, x.size(-2), x.size(-1))
        x = self.cross_attn_4(x, palette_embedding_repeated)
        c4 = x  # Final output without further downsampling

        return c1, c2, c3, c4

