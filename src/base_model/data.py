import json
import cv2
import numpy as np
from skimage.color import rgb2lab


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
        src_palette = np.array(sample["src_palette"])[np.newaxis, :, :]
        src_palette = src_palette[:, :6, :].ravel() / 255.0
        tgt_palette = np.array(sample["tgt_palette"])[np.newaxis, :, :]
        tgt_palette = tgt_palette[:, :6, :].ravel() / 255.0
        return src_image, tgt_image, illu, src_palette, tgt_palette
    

def get_data(dataset_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    data = RecolorizeDataset(json_path=dataset_path, transform=transform)
    return data


# Path to the JSON file
if __name__=="__main__":
    dataset_path = "../../datasets/processed_data_v2/recolorization_data.json"  # Update with your path
    data = get_data(dataset_path)
    # Get a sample from the dataset and visualize
    src_image, tgt_image, illu, src_palette, tgt_palette = data[5]
