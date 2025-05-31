# test_model_benchmark.py

import pytest
import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import sys
sys.path.insert(0, "../")
from model import get_model

def load_model():
    model = get_model()
    model_path = "../checkpoint_epoch_95.pt"
    modelfile = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(modelfile['model_state_dict'])
    model.eval()
    return model

def create_palette_image(palette):
    palette_image = np.zeros((4, 24, 3), dtype=np.float32)
    for i, color in enumerate(palette):
        row = (i // 6) * 4
        col = (i % 6) * 4
        color_lab = rgb2lab(np.array(color, dtype=np.float32).reshape(1, 1, 3) / 255.0).flatten()
        color_lab[0] /= 100
        color_lab[1] = (color_lab[1] + 128) / 256
        color_lab[2] = (color_lab[2] + 128) / 256
        palette_image[row:row + 4, col:col + 4, :3] = color_lab
    return torch.from_numpy(palette_image).permute(2, 0, 1)

def preprocess_image(src_image):
    h, w = src_image.size
    max_dim = 256
    resized_img = src_image.resize((max_dim, max_dim), Image.LANCZOS)
    lab_image = rgb2lab(np.array(resized_img) / 255)
    lab_image[:, :, 0] /= 100
    lab_image[:, :, 1:] = (lab_image[:, :, 1:] + 128) / 256
    illu = torch.from_numpy(lab_image[:, :, 0]).float()
    lab_image = torch.from_numpy(lab_image).permute(2, 0, 1).float()
    return illu, lab_image

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
    out_rgb_pil = Image.fromarray(out_rgb.astype(np.uint8)).resize(src_image.size, Image.LANCZOS)
    return out_rgb_pil

@pytest.fixture(scope='module')
def model():
    return load_model()

def test_model_benchmark(benchmark, model):
    # Load the input image
    src_image = Image.open("../test_images/input.png").convert("RGB")
    palette = [
        [255, 255, 255],
        [255, 221, 174],
        [255, 221, 174],
        [198, 231, 255],
        [212, 246, 255],
        [251, 251, 251]
    ]

    # Prepare inputs
    illu, src_lab_image = preprocess_image(src_image)
    tgt_palette = create_palette_image(palette)
    src_lab_image = src_lab_image.unsqueeze(0)
    illu = illu.unsqueeze(0)
    tgt_palette = tgt_palette.unsqueeze(0)

    def run_model():
        with torch.no_grad():
            output = model(src_lab_image, tgt_palette, illu)
        out_rgb_pil = post_process(output, src_image)

    # Use the benchmark fixture to measure the execution time
    benchmark(run_model)
