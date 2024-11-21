import torch
import torch.nn.functional as F
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by the number of heads"

        # Weight matrices for query, key, and value
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.W_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)  # (b, h*w, c)

        # Project to queries, keys, and values
        q = self.W_q(x)  # (b, h*w, embed_dim)
        k = self.W_k(x)    # (b, h*w, embed_dim)
        v = self.W_v(x)  # (b, h*w, embed_dim)

        # Split into multiple heads and compute scaled dot-product attention
        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (b, num_heads, h*w, head_dim)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (b, num_heads, h*w, head_dim)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (b, num_heads, h*w, head_dim)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (b, num_heads, h*w, h*w)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (b, num_heads, h*w, h*w)
        attn_output = torch.matmul(attn_weights, v)  # (b, num_heads, h*w, head_dim)

        # Combine heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, -1, self.embed_dim)  # (b, h*w, embed_dim)
        attn_output = self.W_out(attn_output)  # (b, h*w, embed_dim)

        # Reshape back to original spatial dimensions
        attn_output = attn_output.permute(0, 2, 1).reshape(b, c, h, w)  # (b, c, h, w)
        return x.permute(0, 2, 1).reshape(b, c, h, w) + attn_output
    

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, palette_embed, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(palette_embed, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_out = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5


    def forward(self, x, palette_embedding):
        # Reshape and project for key and value
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(2, 0, 1)  # Shape: (h * w, b, c)
        k = self.W_k(x_flat)  # Key projection
        v = self.W_v(x_flat)  # Value projection

        # Expand palette_embedding and project for query
        b_, c_, h_, w_ = palette_embedding.shape
        palette_embedding = palette_embedding.view(b_, c_, h_ * w_).permute(2, 0, 1)
        q = self.W_q(palette_embedding)

        # Apply multi-head attention with separate keys and values
        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (b, num_heads, h*w, head_dim)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (b, num_heads, h*w, head_dim)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (b, num_heads, h*w, head_dim)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (b, num_heads, h*w, h*w)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (b, num_heads, h*w, h*w)
        attn_output = torch.matmul(attn_weights, v)  # (b, num_heads, h*w, head_dim)

        # Combine heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, -1, self.embed_dim)  # (b, h*w, embed_dim)
        attn_output = self.W_out(attn_output)  # (b, h*w, embed_dim)

        # Reshape back to original spatial dimensions
        attn_output = attn_output.permute(0, 2, 1).reshape(b, c, h, w)  # (b, c, h, w)
        return x + attn_output  # Concatenate along the channel dimension