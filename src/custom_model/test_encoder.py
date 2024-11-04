import torch
from encoder import RecoloringEncoder  # Make sure this matches the file name

if __name__ == "__main__":
    # Define input dimensions
    batch_size = 4
    height, width = 128, 128  # Adjust these dimensions as needed
    palette_height, palette_width = 4, 60  # Dimensions for the target palettes

    # Instantiate the model
    model = RecoloringEncoder(in_channels=3, palette_embedding_dim=64, num_heads=4)

    # Generate random input tensors
    x = torch.randn(batch_size, 3, height, width)  # RGB image tensor with shape (batch_size, 3, height, width)
    target_palettes = torch.randn(batch_size, 4, palette_height, palette_width)  # Palette tensor with shape (batch_size, 4, 4, 60)

    # Run the model
    c1, c2, c3, c4 = model(x, target_palettes)

    # Print the shapes of each output to verify
    print("Output shapes:")
    print("c1 shape:", c1.shape)  # Expected: (batch_size, 128, height/2, width/2)
    print("c2 shape:", c2.shape)  # Expected: (batch_size, 256, height/4, width/4)
    print("c3 shape:", c3.shape)  # Expected: (batch_size, 256, height/8, width/8)
    print("c4 shape:", c4.shape)  # Expected: (batch_size, 256, height/16, width/16)
