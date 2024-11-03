import torch
from decoder import RecoloringDecoder


if __name__=="__main__":
    # Define input dimensions
    batch_size = 4
    height, width = 32, 32  # Adjust as needed based on your model's requirements
    palette_height, palette_width = 4, 60  # Dimensions for the target_palettes image

    # Instantiate the model
    model = RecoloringDecoder(palette_embedding_dim=64, num_heads=4)

    # Generate random input tensors
    c1 = torch.randn(batch_size, 512, height // 8, width // 8)  # Shape: (batch_size, 256, h/8, w/8)
    c2 = torch.randn(batch_size, 256, height // 4, width // 4)  # Shape: (batch_size, 256, h/4, w/4)
    c3 = torch.randn(batch_size, 128, height // 2, width // 2)  # Shape: (batch_size, 128, h/2, w/2)
    c4 = torch.randn(batch_size, 64, height, width)             # Shape: (batch_size, 64, h, w)
    target_palettes = torch.randn(batch_size, 4, palette_height, palette_width)  # Shape: (batch_size, 4, 16, 4)
    illu = torch.randn(batch_size, height, width)  # Illumination map, shape: (batch_size, h, w)

    # Run the model
    output = model(c1, c2, c3, c4, target_palettes, illu)

    # Print the output shape
    print("Output shape:", output.shape)

