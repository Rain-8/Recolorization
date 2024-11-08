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
from PIL import Image


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
        self.width = 256
        self.height = 256

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
        src_image = cv2.resize(src_image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        tgt_image = cv2.imread(sample["tgt_image_path"])
        tgt_image = cv2.cvtColor(tgt_image, cv2.COLOR_BGR2RGB)
        tgt_image = cv2.resize(tgt_image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        src_image_lab = rgb2lab(src_image/255)  # Convert to HxWxC for skimage
        tgt_image_lab = rgb2lab(tgt_image/255)
        src_image_lab[:, :, 0] /= 100
        src_image_lab[:, :, 1] = (src_image_lab[:, :, 1] + 128)/ 256 
        src_image_lab[:, :, 2] = (src_image_lab[:, :, 2] + 128)/ 256 
        tgt_image_lab[:, :, 0] /= 100
        tgt_image_lab[:, :, 1] = (tgt_image_lab[:, :, 1] + 128)/ 256 
        tgt_image_lab[:, :, 2] = (tgt_image_lab[:, :, 2] + 128)/ 256 

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
            color_lab[0] /= 100
            color_lab[1] = (color_lab[1] + 128)/ 256 
            color_lab[2] = (color_lab[2] + 128)/ 256 
    
            # Place LAB values in the RGB channels of the palette image
            palette_image[row:row+4, col:col+4, :3] = color_lab  # LAB channels
    
        # Convert to torch tensor and reorder to (channels, height, width)
        return torch.from_numpy(palette_image).permute(2, 0, 1)  # Shape (4, 4, 24)
    
    
    def get_data(dataset_path, sample=None):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        data = RecolorizeDataset(json_path=dataset_path, sample=sample)
        return data
    
    
def visualize_and_save_recolor_data(src_image, tgt_image, illu, src_palette, tgt_palette, output_dir="output"):
    """
    Save components of the recoloring dataset (source image, target image, palettes, and illumination)
    as individual files, assuming all inputs are in LAB format.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    
    def tensor_to_image(tensor):
        if isinstance(tensor, torch.Tensor):
            if tensor.dim() == 3:
                return tensor.permute(1, 2, 0).numpy()
            return tensor.numpy()
        return tensor

    # Convert LAB images to RGB for saving
    src_image = np.clip(tensor_to_image(src_image), 0, 100)
    tgt_image = np.clip(tensor_to_image(tgt_image), 0, 100)
    illu_np = tensor_to_image(illu)

    # Normalize and prepare palettes in LAB format
    src_palette_lab = np.clip(tensor_to_image(src_palette)[:, :, :3], 0, 100)
    tgt_palette_lab = np.clip(tensor_to_image(tgt_palette)[:, :, :3], 0, 100)

    # Convert LAB to RGB
    src_image_rgb = (lab2rgb(src_image) * 255).astype(np.uint8)
    tgt_image_rgb = (lab2rgb(tgt_image) * 255).astype(np.uint8)
    src_palette_rgb = (lab2rgb(src_palette_lab) * 255).astype(np.uint8)
    tgt_palette_rgb = (lab2rgb(tgt_palette_lab) * 255).astype(np.uint8)
    illu_rgb = (illu_np * 255).astype(np.uint8)

    # Save images using PIL
    Image.fromarray(src_image_rgb).save(f"{output_dir}/source_image.png")
    Image.fromarray(tgt_image_rgb).save(f"{output_dir}/target_image.png")
    Image.fromarray(src_palette_rgb).save(f"{output_dir}/source_palette.png")
    Image.fromarray(tgt_palette_rgb).save(f"{output_dir}/target_palette.png")
    Image.fromarray(illu_rgb).save(f"{output_dir}/illumination_map.png")

    print(f"Images saved in directory: {output_dir}")


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
    dataset_path = "../../datasets/processed_palettenet_data_sample_v4/val.json"  # Update with your path
    data = get_data(dataset_path)

    # Get a sample from the dataset and visualize
    src_image, tgt_image, illu, src_palette, tgt_palette = data[6]
    visualize_and_save_recolor_data(src_image, tgt_image, illu, src_palette, tgt_palette)
