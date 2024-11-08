import json
import cv2
import numpy as np
import random
from skimage.color import rgb2lab
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.color import lab2rgb


def get_illuminance(img):
    """
    Get the luminance of an image. Shape: (h, w)
    """
    img = img.permute(1, 2, 0)  # (h, w, channel) 
    img = img.numpy()
    img = img.astype(np.float32) / 255.0
    img_L = img[:, :, 0]  # luminance (h, w)
    return torch.from_numpy(img_L)


class RecolorizeDataset(Dataset):
    def __init__(self, json_path, transform=None, sample=None):
        super().__init__()
        self.transform = transform
        # Load the JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        if sample is not None:
            self.data = random.sample(self.data, k=sample)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        # Get keys to access the JSON dictionary in a stable order
        keys = list(self.data.keys())
        sample_key = keys[idx]
        sample = self.data[sample_key]
    
        # Load images in RGB format
        src_image = cv2.imread(sample["src_image_path"])
        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        tgt_image = cv2.imread(sample["tgt_image_path"])
        tgt_image = cv2.cvtColor(tgt_image, cv2.COLOR_BGR2RGB)
    
        # Apply transformations if any
        if self.transform:
            src_image = self.transform(src_image)
            tgt_image = self.transform(tgt_image)
    
        # Convert images to LAB color space
        src_image_lab = rgb2lab(src_image.permute(1, 2, 0).numpy()/255)  # Convert to HxWxC for skimage
        tgt_image_lab = rgb2lab(tgt_image.permute(1, 2, 0).numpy()/255)
    
        # Extract the L (luminance) channel for illuminance from source image in LAB
        illu = torch.from_numpy(src_image_lab[:, :, 0]).float()  # L channel only, as a tensor
    
        # Convert LAB images back to torch tensors for consistent return types
        src_image_lab = torch.from_numpy(src_image_lab).permute(2, 0, 1).float()  # CxHxW
        tgt_image_lab = torch.from_numpy(tgt_image_lab).permute(2, 0, 1).float()  # CxHxW
    
        # Create palettes in 4x60x4 format, assuming palettes are stored in RGB
        src_palette = self.create_palette_image(sample["src_palette"], convert_to_lab=True)
        tgt_palette = self.create_palette_image(sample["tgt_palette"], convert_to_lab=True)
    
        return src_image_lab, tgt_image_lab, illu, src_palette, tgt_palette
    
   
    
    def create_palette_image(self, palette, convert_to_lab=True):
        """
        Create a 4x24x4 palette image from a palette list, with colors in LAB format. 
        Each color occupies a 4x4 block. The alpha channel is set to 1 for existing colors and 0 for non-existing slots.
        """
        palette_image = np.zeros((4, 24, 3), dtype=np.float32)  # Initialize with zeros (RGBA)
    
        for i, color in enumerate(palette):
            row = (i // 6) * 4  # Each row contains 6 colors, starting at row 0 or 4
            col = (i % 6) * 4   # Each color occupies a 4x4 block
    
            # Convert RGB color to LAB and normalize
            color_lab = rgb2lab(np.array(color, dtype=np.float32).reshape(1, 1, 3) / 255.0).flatten()
            color_lab[0] /= 100.0  # Normalize L* to [0, 1]
            color_lab[1] = (color_lab[1] + 128) / 255.0  # Normalize a* to [0, 1]
            color_lab[2] = (color_lab[2] + 128) / 255.0  # Normalize b* to [0, 1]
    
            # Place LAB values in the RGB channels of the palette image
            palette_image[row:row+4, col:col+4, :3] = color_lab  # LAB channels
            # palette_image[row:row+4, col:col+4, 3] = 1           # Alpha channel set to 1
    
        # Convert to torch tensor and reorder to (channels, height, width)
        return torch.from_numpy(palette_image).permute(2, 0, 1)  # Shape (4, 4, 24)
    
    
    def get_data(dataset_path, sample=None):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        data = RecolorizeDataset(json_path=dataset_path, transform=transform, sample=sample)
        return data
    
    
def visualize_recolor_data(src_image, tgt_image, illu, src_palette, tgt_palette, figsize=(20, 10)):
    """
    Visualize all components of the recoloring dataset in a single figure, 
    assuming all inputs are in LAB format, with LAB conversion and luminance adjustment.
    """
    def tensor_to_image(tensor):
        if isinstance(tensor, torch.Tensor):
            if tensor.dim() == 3:
                return tensor.permute(1, 2, 0).numpy()
            return tensor.numpy()
        return tensor

    # Ensure LAB values are in the correct range before conversion
    src_image_lab[0] *= 100  # Scale L* back to [0, 100]
    src_image_lab[1] = src_image_lab[1] * 255 - 128  # Scale a* back to [-128, 127]
    src_image_lab[2] = src_image_lab[2] * 255 - 128  # Scale b* back to [-128, 127]
    
    tgt_image_lab[0] *= 100
    tgt_image_lab[1] = tgt_image_lab[1] * 255 - 128
    tgt_image_lab[2] = tgt_image_lab[2] * 255 - 128
    
    src_palette[0] *= 100
    src_palette[1] = src_palette[1] * 255 - 128
    src_palette[2] = src_palette[2] * 255 - 128
    
    tgt_palette[0] *= 100
    tgt_palette[1] = tgt_palette[1] * 255 - 128
    tgt_palette[2] = tgt_palette[2] * 255 - 128
    
    # Convert LAB images and palettes to RGB
    src_image_rgb = np.clip(lab2rgb(src_image_lab), 0, 1)
    tgt_image_rgb = np.clip(lab2rgb(tgt_image_lab), 0, 1)
    src_palette_rgb = np.clip(lab2rgb(src_palette), 0, 1)
    tgt_palette_rgb = np.clip(lab2rgb(tgt_palette), 0, 1)


    illu_np = tensor_to_image(illu)

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
    ax1.imshow(src_image_rgb)
    ax1.set_title("Source Image", fontsize=12, pad=10)
    ax1.axis("off")

    ax2.imshow(tgt_image_rgb)
    ax2.set_title("Target Image (Luminance Matched)", fontsize=12, pad=10)
    ax2.axis("off")

    # Plot difference map in LAB
    diff = np.abs(src_image_lab - tgt_image_lab).mean(axis=-1)
    im3 = ax3.imshow(diff, cmap='hot')
    ax3.set_title("LAB Color Difference Map", fontsize=12, pad=10)
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Plot palettes
    ax4.imshow(src_palette_rgb)
    ax4.set_title("Source Palette", fontsize=12, pad=10)
    ax4.axis("off")

    ax5.imshow(tgt_palette_rgb)
    ax5.set_title("Target Palette", fontsize=12, pad=10)
    ax5.axis("off")

    # Plot illumination map
    im6 = ax6.imshow(illu_np, cmap='gray')
    ax6.set_title("Illumination Map", fontsize=12, pad=10)
    ax6.axis("off")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def get_data(dataset_path, sample=None):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    data = RecolorizeDataset(json_path=dataset_path, transform=transform, sample=sample)
    return data

if __name__=="__main__":
    # Path to the JSON file
    dataset_path = "../../datasets/processed_palettenet_data_sample_v3/val.json"  # Update with your path
    data = get_data(dataset_path)

    # Get a sample from the dataset and visualize
    src_image, tgt_image, illu, src_palette, tgt_palette = data[9]
    fig = visualize_recolor_data(src_image, tgt_image, illu, src_palette, tgt_palette)
    plt.show()
