import json
import random
import cv2
import numpy as np
from skimage.color import rgb2lab


import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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
        src_palette = np.array(sample["src_palette"])[np.newaxis, :, :]
        src_palette = src_palette[:, :6, :].ravel() / 255.0
        tgt_palette = np.array(sample["tgt_palette"])[np.newaxis, :, :]
        tgt_palette = tgt_palette[:, :6, :].ravel() / 255.0
        return src_image, tgt_image, illu, src_palette, tgt_palette
    

def get_data(dataset_path, sample=None):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    data = RecolorizeDataset(json_path=dataset_path, transform=transform, sample=sample)
    return data


def visualize_recolor_data(src_image, tgt_image, illu, figsize=(20, 10)):
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
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 4, figure=fig)
    
    # Main images (larger)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])

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

    # Plot illumination map
    im6 = ax4.imshow(illu_np, cmap='gray')
    ax4.set_title("Illumination Map", fontsize=12, pad=10)
    ax4.axis("off")
    plt.colorbar(im6, ax=ax4, fraction=0.046, pad=0.04)

    # Adjust layout
    plt.tight_layout()
    return fig

# Path to the JSON file
if __name__=="__main__":
    dataset_path = "../../datasets/processed_palettenet_data_sample/test/recolorization_data.json"  # Update with your path
    data = get_data(dataset_path)
    # Get a sample from the dataset and visualize
    src_image, tgt_image, illu, src_palette, tgt_palette = data[250]
    fig = visualize_recolor_data(src_image, tgt_image, illu)
    plt.show()
