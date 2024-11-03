import json
import cv2
import numpy as np
from skimage.color import rgb2lab
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_illuminance(img):
    """
    Get the luminance of an image. Shape: (h, w)
    """
    img = img.permute(1, 2, 0)  # (h, w, channel) 
    img = img.numpy()
    img = img.astype(np.float32) / 255.0
    img_LAB = rgb2lab(img)
    img_L = img_LAB[:, :, 0]  # luminance (h, w)
    return torch.from_numpy(img_L)


class RecolorizeDataset(Dataset):
    def __init__(self, json_path, transform=None):
        super().__init__()
        self.transform = transform
        # Load the JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get keys to access the JSON dictionary in a stable order
        keys = list(self.data.keys())
        sample_key = keys[idx]
        sample = self.data[sample_key]

        # Load images
        src_image = cv2.imread(sample["src_image_path"])
        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        tgt_image = cv2.imread(sample["tgt_image_path"])
        tgt_image = cv2.cvtColor(tgt_image, cv2.COLOR_BGR2RGB)
        if self.transform:
            src_image = self.transform(src_image)
            tgt_image = self.transform(tgt_image)

        # Calculate illuminance from the source image
        illu = get_illuminance(src_image)

        # Create palettes in 4x60x4 format
        src_palette = self.create_palette_image(sample["src_palette"])
        tgt_palette = self.create_palette_image(sample["tgt_palette"])
        return src_image, tgt_image, illu, src_palette, tgt_palette

    def create_palette_image(self, palette):
        """
        Create a 4x60x4 palette image from a palette list. Each color occupies a 4x4 block.
        The alpha channel is set to 1 for existing colors and 0 for non-existing slots.
        """
        palette_image = np.zeros((4, 60, 4), dtype=np.float32)  # Initialize with zeros (RGBA)

        for i, color in enumerate(palette):
            row = (i // 15) * 4  # Each row contains 15 colors, starting at row 0 or 4
            col = (i % 15) * 4   # Each color occupies a 4x4 block
            palette_image[row:row+4, col:col+4, :3] = np.array(color, dtype=np.float32) / 255  # RGB channels
            palette_image[row:row+4, col:col+4, 3] = 1     # Alpha channel set to 1

        return torch.from_numpy(palette_image).permute(2, 0, 1)  # Convert to tensor (4, 4, 60)


def get_data(dataset_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    data = RecolorizeDataset(json_path=dataset_path, transform=transform)
    return data


def visualize_recolor_data(src_image, tgt_image, illu, src_palette, tgt_palette, figsize=(20, 10)):
    """
    Visualize all components of the recoloring dataset in a single figure.
    """
    # Convert tensors to numpy arrays for visualization
    def tensor_to_image(tensor):
        if isinstance(tensor, torch.Tensor):
            if tensor.dim() == 3:
                return tensor.permute(1, 2, 0).numpy()
            return tensor.numpy()
        return tensor

    src_image_np = np.clip(tensor_to_image(src_image), 0, 1)
    tgt_image_np = np.clip(tensor_to_image(tgt_image), 0, 1)
    illu_np = tensor_to_image(illu)
    src_palette_np = np.clip(tensor_to_image(src_palette)[:, :, :3], 0, 255)  # Only RGB channels
    tgt_palette_np = np.clip(tensor_to_image(tgt_palette)[:, :, :3], 0, 255)  # Only RGB channels
    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig)
    
    # Main images (larger)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Palettes and illumination (smaller)
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # Plot images
    ax1.imshow(src_image_np)
    ax1.set_title("Source Image", fontsize=12, pad=10)
    ax1.axis("off")

    ax2.imshow(tgt_image_np)
    ax2.set_title("Target Image", fontsize=12, pad=10)
    ax2.axis("off")

    # Plot difference map
    diff = np.abs(src_image_np - tgt_image_np).mean(axis=-1)
    im3 = ax3.imshow(diff, cmap='hot')
    ax3.set_title("Color Difference Map", fontsize=12, pad=10)
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Plot palettes
    ax4.imshow(src_palette_np)
    ax4.set_title("Source Palette", fontsize=12, pad=10)
    ax4.axis("off")

    ax5.imshow(tgt_palette_np)
    ax5.set_title("Target Palette", fontsize=12, pad=10)
    ax5.axis("off")

    # Plot illumination map
    im6 = ax6.imshow(illu_np, cmap='gray')
    ax6.set_title("Illumination Map", fontsize=12, pad=10)
    ax6.axis("off")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    # Adjust layout
    plt.tight_layout()
    return fig


if __name__=="__main__":
    # Path to the JSON file
    dataset_path = "../../datasets/processed_data_v2/recolorization_data.json"  # Update with your path
    data = get_data(dataset_path)

    # Get a sample from the dataset and visualize
    src_image, tgt_image, illu, src_palette, tgt_palette = data[5]
    fig = visualize_recolor_data(src_image, tgt_image, illu, src_palette, tgt_palette)
    plt.show()