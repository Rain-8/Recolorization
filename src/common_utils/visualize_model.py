import os
import sys

import torch
from torchviz import make_dot

sys.path.insert(0, "../custom_model/")
from model import get_model  # Adjust the import according to your file structure

# Create an instance of the model
model = get_model()

# Create dummy input tensors with appropriate shapes
ori_img = torch.randn(1, 3, 256, 256)  # Example input image with shape (batch_size, channels, height, width)
tgt_palette = torch.randn(1, 4 * 24 * 3)  # Example target palette
illu = torch.randn(1, 256, 256)  # Example illumination input

# Forward pass through the model to get the output and create a computation graph
output = model(ori_img, tgt_palette, illu)

# Generate the graph
dot = make_dot(output, params=dict(model.named_parameters()))

# Save the graph as a PDF or PNG file
os.makedirs("../../assets/", exist_ok=True)
dot.render("../../assets/recolorizer_model", format="png")  # Change format as needed