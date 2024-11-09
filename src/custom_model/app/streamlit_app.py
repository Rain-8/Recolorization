import torch
import numpy as np
import streamlit as st
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from model import get_model
import torch.nn.functional as F

def preprocess_image(src_image):
    h = src_image.size[0]
    w = src_image.size[1]
    max_dim = 256
    # if h > w:
    #     # Height is the larger dimension
    #     new_h = max_dim
    #     new_w = int(max_dim * (w / h))
    # else:
    #     # Width is the larger dimension
    #     new_w = max_dim
    #     new_h = int(max_dim * (h / w))
    resized_img = src_image.resize((max_dim, max_dim), Image.LANCZOS)
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


def load_model(model_path):
    # this has the model architecture
    model = get_model()
    modelfile = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(modelfile['model_state_dict'])
    model.eval()
    return model


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


st.title("Image Recoloring App")
model_path = "checkpoint_epoch_90.pt" 
model = load_model(model_path)
# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load the image
    rgb_image = Image.open(uploaded_file).convert("RGB")
    st.image(rgb_image, caption='Input RGB Image', use_column_width=True)
    colors = []
    cols = st.columns(6)   # Create six columns for six colors
    
    for i in range(6):
        with cols[i]:   # Use each column context for the color picker
            color = st.color_picker(f"Pick Color {i + 1}", "#FFFFFF")  
            colors.append([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]) 

    illu, src_lab_image = preprocess_image(rgb_image)
    tgt_palette = create_palette_image(colors)
    src_lab_image = src_lab_image.unsqueeze(0)
    illu = illu.unsqueeze(0)
    tgt_palette = tgt_palette.unsqueeze(0)
    
    if st.button("Recolor Image"):
        with torch.no_grad():
            output = model(src_lab_image, tgt_palette, illu)
        out_rgb_pil = post_process(output, rgb_image)
        st.image(out_rgb_pil, caption='Output RGB Image', use_column_width=True)
