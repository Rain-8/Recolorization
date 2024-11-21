import os
import json
from turtle import mode
import torch
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from PIL import Image

from model import get_model

def load_model():
    # this has the model architecture
    model = get_model()
    model_path = "checkpoint_epoch_90.pt" 
    modelfile = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(modelfile['model_state_dict'])
    model.eval()
    return model


def create_palette_image(palette, convert_to_lab=True):
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


def preprocess_image(src_image):
    h = src_image.size[0]
    w = src_image.size[1]
    max_dim = 400
    if h > w:
        new_h = max_dim
        new_w = int(max_dim * (w / h))
    else:
        new_w = max_dim
        new_h = int(max_dim * (h / w))
    new_h = 16 * (new_h // 16)
    new_w = 16 * (new_w // 16)
    resized_img = src_image.resize((new_h, new_w), Image.LANCZOS)
    lab_image = rgb2lab(np.array(resized_img)/255)
    lab_image[:, :, 0] = lab_image[:, :, 0]/100
    lab_image[:, :, 1:] = (lab_image[:, :, 1:] + 128)/ 256
    illu = torch.from_numpy(lab_image[:, :, 0]).float()  # L channel only, as a tensor
    lab_image = torch.from_numpy(lab_image).permute(2, 0, 1).float()  # CxHxW
    return illu, lab_image


def create_palette_image(palette):
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


def tensor_to_image(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.dim() == 3:
            return tensor.permute(1, 2, 0).detach().numpy()
        return tensor.detach().numpy()
    return tensor


def post_process(output, src_image):
    out_np = tensor_to_image(output.squeeze(0))
    out_np[:, :, 0] = np.clip(out_np[:, :, 0], 0, 1) * 100
    out_np[:, :, 1:] = np.clip(out_np[:, :, 1:], 0, 1) * 255 - 128
    out_rgb = np.clip(lab2rgb(out_np) * 255, 0, 255)
    out_rgb_pil = Image.fromarray(out_rgb.astype(np.uint8)).resize((src_image.size[0], src_image.size[1]), Image.LANCZOS)
    return out_rgb_pil

def get_images_and_palettes(json_file_path):
    # Load the JSON data
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Create a list to store the results
    results = []

    # Iterate over the data
    for key, value in data.items():
        src_image_path = value["src_image_path"]
        tgt_palette = value["tgt_palette"]

        # Add the source image path and target palette to the results list
        results.append((src_image_path, tgt_palette))

    return results

def test_model():
    tgt_dir = "Test_results/"
    test_data = get_images_and_palettes("test_model.json")
    print("Length of test data:", len(test_data))
    model = load_model()
    results = {}
    for src_image_path, palette in test_data[:10]:
        src_image = Image.open(src_image_path).convert("RGB")
        illu, src_lab_image = preprocess_image(src_image)
        tgt_palette = create_palette_image(palette)
        src_lab_image = src_lab_image.unsqueeze(0)
        illu = illu.unsqueeze(0)
        tgt_palette = tgt_palette.unsqueeze(0)
        with torch.no_grad():
            output = model(src_lab_image, tgt_palette, illu)
        out_rgb_pil = post_process(output, src_image)
        out_palette = post_process(tgt_palette, src_image)
        os.makedirs(tgt_dir, exist_ok=True)
        out_rgb_pil.save(tgt_dir + src_image_path.split("/")[-1])
        out_palette.save(tgt_dir + src_image_path.split("/")[-1].split(".")[0] + "_palette.png")
        results[src_image_path] = {
            "tgt_image_path": tgt_dir + src_image_path.split("/")[-1],
            "pal_image_path": "palettenet_images/" + src_image_path.split("/")[-1],
            "tgt_palette": palette
        }
    with open(tgt_dir + "test_meta.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to", tgt_dir + "test_meta.json")




if __name__ == "__main__":
    test_model()
