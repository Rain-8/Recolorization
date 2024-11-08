import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from model import get_model
from skimage.color import rgb2lab
from skimage.color import lab2rgb


# Load your trained model function (update this to your actual model loading function)
def load_model(model_path):
    # this has the model architecture
    model = get_model()
    # print(model)
    model_path = "checkpoint_epoch_55.pt" 
    modelfile = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(modelfile['model_state_dict'])
    # weights
    # print(model.state_dict())

    model.eval()
    # inputs source image, target apllete, illuminance
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

def rgb_to_lab(rgb_image):
    """
    Convert an RGB image to LAB color space.
    
    Args:
        rgb_image (PIL.Image): The input RGB image.
    
    Returns:
        np.ndarray: The LAB image.
    """
    # Convert the RGB image to LAB
    lab_image = rgb_image.convert('LAB')
    
    # Convert the LAB image to a numpy array
    lab_array = np.array(lab_image)
    
    return lab_array

def normalize_layer(layer):
    """
    Normalize the LAB values to the range [0, 1].
    
    Args:
        lab_array (np.ndarray): The input LAB image.
    
    Returns:
        np.ndarray: The normalized LAB image.
    """
    # Normalize the L, a, and b channels separately
    min_layer = layer.min()
    max_layer = layer.max()
    norm_layer = (layer - min_layer) / (max_layer - min_layer)
    
    print("Norm Layer:", norm_layer)

    return norm_layer

def lab_to_rgb(lab_array):
    """
    Convert a LAB image to RGB color space.
    
    Args:
        lab_array (np.ndarray): The input LAB image.
    
    Returns:
        PIL.Image: The reconstructed RGB image.
    """
    lab_image = Image.fromarray(np.uint8(lab_array * 255), mode='LAB')
    # st.image(lab_image, caption='LAB', use_column_width=True)
    # Convert the LAB image to RGB
    rgb_image = lab_image.convert('RGB')
    
    return rgb_image


# Preprocess image function (similar to what you have)
def preprocess_image(image):
    st.image(image, caption='Image', use_column_width=True) # HERE IT IS FINE
    resized_img = image.resize((256, 256), Image.LANCZOS)
    st.image(resized_img, caption='resize', use_column_width=True)
    lab_img = rgb_to_lab(resized_img)
    st.image(lab_img, caption='to lab', use_column_width=True)
    
    # Normalize the LAB values
    norm_lab_img = np.zeros_like(lab_img, dtype=np.float32)
    norm_lab_img[:, :, 0] = normalize_layer(lab_img[:, :, 0])
    norm_lab_img[:, :, 1] = normalize_layer(lab_img[:, :, 1])
    norm_lab_img[:, :, 2] = normalize_layer(lab_img[:, :, 2])

    st.image(norm_lab_img, caption='normalized', use_column_width=True)
    
    return norm_lab_img # Add batch dimension

# Function to recolor the image (this should call your model)
def recolor_image(model, src_image, tgt_palette):
    with torch.no_grad():
        # print("TGT PALETTER:", tgt_palette)
        # Extract the L (luminance) channel for illuminance from source image in LAB
        illu = torch.from_numpy(src_image[:, :, 0]).float()  # L channel only, as a tensor
        src_image_tensor = torch.from_numpy(src_image).permute(2, 0, 1).float() 
        src_image_tensor = src_image_tensor.unsqueeze(0)  # Add batch dimension

        print("SRC IMAGE TENSOR:", src_image_tensor.size())
        print("TGT PALETTE:", tgt_palette.size())

        # output = src_image_tensor

        output = model(src_image_tensor, tgt_palette.flatten(), illu.unsqueeze(0))  # Add batch dimension for illu
        print("OUTPUT:", output)
        # print("Source Image Tensor - L channel range:", src_image_tensor[0, 0, :, :].min().item(), src_image_tensor[0, 0, :, :].max().item())
        # print("Source Image Tensor - a channel range:", src_image_tensor[0, 1, :, :].min().item(), src_image_tensor[0, 1, :, :].max().item())
        # print("Source Image Tensor - b channel range:", src_image_tensor[0, 2, :, :].min().item(), src_image_tensor[0, 2, :, :].max().item())
    return output

def tensor_to_image(tensor):
        if isinstance(tensor, torch.Tensor):
            if tensor.dim() == 3:
                return tensor.permute(1, 2, 0).numpy()
            return tensor.numpy()
        return tensor

# Streamlit app layout
st.title("Image Recoloring App")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load the image
    rgb_image = Image.open(uploaded_file)

    # Create six color pickers for the new palette
    colors = []
    for i in range(6):
        color = st.color_picker(f"Pick Color {i + 1}", "#FFFFFF")  # Default to white
        colors.append([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)])  # Convert hex to RGB

    tgt_palette = np.array(colors)  # Create a palette from selected colors
    tgt_palette = create_palette_image(tgt_palette, convert_to_lab=True)
    tgt_palette = tgt_palette.unsqueeze(0)  # Add batch dimension

    if st.button("Recolor Image"):
        # Load the model (make sure to specify the correct path)
        model_path = "path_to_your_model.pt"  # Update with your actual model path
        model = load_model(model_path)

        # Preprocess the uploaded image
        src_image_array = preprocess_image(rgb_image)

        # Recolor the image using the selected palette
        output_image_tensor = recolor_image(model, src_image_array, tgt_palette)
        output_image_tensor= output_image_tensor.squeeze(0)
        print("output_image_tensor:", output_image_tensor.size())

        # output_image_np is normalized lab image, convert to 0-255 RGB
        output_image_np = output_image_tensor.permute(1, 2, 0).numpy()

        print("PLS BE THE SAME", np.allclose(src_image_array, output_image_np))

        # Denormalize the LAB image
        #Compare src_image_array and output_image_np if they are the same 


        output_image = lab_to_rgb(output_image_np)
        st.image(output_image, caption='Output RGB Image', use_column_width=True)
        # output_image_np[:, :, 0] = output_image_np[:, :, 0] * 100  # L channel range [0, 100]
        # output_image_np[:, :, 1:] = output_image_np[:, :, 1:] * 255 - 128  # a, b channel range [-128, 127]

        # print("Denormalized Output - L channel range:", output_image_np[:, :, 0].min(), output_image_np[:, :, 0].max())
        # print("Denormalized Output - a channel range:", output_image_np[:, :, 1].min(), output_image_np[:, :, 1].max())
        # print("Denormalized Output - b channel range:", output_image_np[:, :, 2].min(), output_image_np[:, :, 2].max())

        # Convert LAB to RGB using OpenCV
        # output_image_np = output_image_np.astype(np.uint8)
        # rgb_image = cv2.cvtColor(output_image_np, cv2.COLOR_LAB2RGB)
        # display_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        # input_image = Image.open(uploaded_file)
        # st.image(input_image, caption='Input RGB Image', use_column_width=True)
        # lab_image = input_image.convert('LAB')
        # # print(np.array(lab_image))
        # rgb_image_converted = lab_image.convert('RGB')
        # st.image(rgb_image_converted, caption='Output RGB Image', use_column_width=True)
        # # COLOR_LAB2RGB
        

        # Display the recolored image
        # st.image(display_image, caption='Recolored Image', use_column_width=True)


