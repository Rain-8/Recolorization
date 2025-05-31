import torch
from encoder import SelfAttention  # Ensure this import points to your actual encoder file

def test_self_attention():
    embed_dim = 16  # Use a very small embed_dim to reduce memory usage
    num_heads = 2

    # Initialize the SelfAttention module with smaller dimensions
    self_attn = SelfAttention(embed_dim=embed_dim, num_heads=num_heads)

    # Use very small input dimensions for testing
    batch_size = 1
    channels = embed_dim  # Set channels to match embed_dim
    height, width = 2, 2  # Minimal spatial dimensions

    # Input tensor with shape (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)

    # Test the self-attention module
    try:
        output = self_attn(x)
        if output is not None:
            print("SelfAttention output shape:", output.shape)
    except RuntimeError as e:
        print("RuntimeError:", e)

if __name__ == "__main__":
    test_self_attention()
