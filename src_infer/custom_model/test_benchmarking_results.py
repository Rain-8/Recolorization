# test_model_benchmark.py

import pytest
import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from model import get_model

import logging



# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_model():
    """Load the model and its weights."""
    model = get_model()
    model_path = "checkpoint_epoch_90.pt"  # Update the path to your model checkpoint
    #modelfile = torch.load(model_path, map_location=device)  # Load model to the correct device
    modelfile = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(modelfile['model_state_dict'])
    model.to(device)  # Move the model to GPU if available
    model.eval()
    return model

def create_palette_image(palette):
    """Create a palette image tensor from a palette list."""
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
    """Preprocess the source image for the model."""
    h, w = src_image.size
    max_dim = 400
    resized_img = src_image.resize((max_dim, max_dim), Image.LANCZOS)
    lab_image = rgb2lab(np.array(resized_img) / 255)
    lab_image[:, :, 0] /= 100
    lab_image[:, :, 1:] = (lab_image[:, :, 1:] + 128) / 256
    illu = torch.from_numpy(lab_image[:, :, 0]).float()
    lab_image = torch.from_numpy(lab_image).permute(2, 0, 1).float()
    return illu, lab_image

def tensor_to_image(tensor):
    """Convert a tensor to an image array."""
    if isinstance(tensor, torch.Tensor):
        if tensor.dim() == 3:
            return tensor.permute(1, 2, 0).detach().cpu().numpy()  # Ensure output is moved to CPU
        return tensor.detach().cpu().numpy()
    return tensor

def post_process(output, src_image):
    """Post-process the model output to obtain the final image."""
    out_np = tensor_to_image(output.squeeze(0))
    out_np[:, :, 0] = np.clip(out_np[:, :, 0], 0, 1) * 100
    out_np[:, :, 1:] = np.clip(out_np[:, :, 1:], 0, 1) * 255 - 128
    out_rgb = np.clip(lab2rgb(out_np) * 255, 0, 255)
    return Image.fromarray(out_rgb.astype(np.uint8)).resize(src_image.size, Image.LANCZOS)

@pytest.fixture(scope='module')
def model():
    """Fixture to load the model."""
    return load_model()

def test_model_benchmark(benchmark, model):
    """Benchmark the model on a single image."""
    # Load the input image
    src_image = Image.open("test_results/img_3.png").convert("RGB")
    palette = [
      [112, 91, 230],
      [117, 134, 133],
      [148, 165, 180],
      [191, 163, 218],
      [230, 160, 197],
      [241, 230, 228]
    ]

    # Prepare inputs
    illu, src_lab_image = preprocess_image(src_image)
    tgt_palette = create_palette_image(palette)
    src_lab_image = src_lab_image.unsqueeze(0).to(device)
    illu = illu.unsqueeze(0).to(device)
    tgt_palette = tgt_palette.unsqueeze(0).to(device)

    # Define the model inference as a function to be benchmarked
    def run_model():
        """Run the model inference."""
        with torch.no_grad():
            # Perform inference
            output = model(src_lab_image, tgt_palette, illu)
            # Post-process the output to get the final RGB image
            out_rgb_pil = post_process(output, src_image)

            # Save the output image for visual inspection
            # output_path = "test_results/benchmark_output/benchmark_result.png"
            # out_rgb_pil.save(output_path)
            # print(f"Output image saved at: {output_path}")

    # Use the benchmark fixture to measure the execution time
    benchmark(run_model)
