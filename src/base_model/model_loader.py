import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from model import get_model
from skimage.color import rgb2lab

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


# Load your trained model function (update this to your actual model loading function)
def load_model(model_path):
    # this has the model architecture
    model = get_model()
    print(model)
    model_path = "checkpoint_epoch_40.pt" 
    modelfile = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(modelfile['model_state_dict'])
    # weights
    print(model.state_dict())

    model.eval()
    # inputs source image, target apllete, illuminance
    return model

# Preprocess image function (similar to what you have)
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to recolor the image (this should call your model)
def recolor_image(model, src_image_tensor, tgt_palette):
    with torch.no_grad():
        illu = get_illuminance(src_image_tensor.squeeze(0))  # Calculate illuminance
        output = model(src_image_tensor, tgt_palette.flatten(), illu.unsqueeze(0))  # Add batch dimension for illu
    return output

# Streamlit app layout
st.title("Image Recoloring App")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption='Original Image', use_column_width=True)

    # Create six color pickers for the new palette
    colors = []
    for i in range(6):
        color = st.color_picker(f"Pick Color {i + 1}", "#FFFFFF")  # Default to white
        colors.append([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)])  # Convert hex to RGB

    tgt_palette = np.array(colors)  # Create a palette from selected colors
    tgt_palette = tgt_palette[np.newaxis, :, :]
    tgt_palette = tgt_palette[:, :6, :].ravel() / 255.0

    if st.button("Recolor Image"):
        # Load the model (make sure to specify the correct path)
        model_path = "path_to_your_model.pt"  # Update with your actual model path
        model = load_model(model_path)

        # Preprocess the uploaded image
        src_image_tensor = preprocess_image(original_image)

        # Recolor the image using the selected palette
        output_image_tensor = recolor_image(model, src_image_tensor, tgt_palette)

        # Convert output tensor back to an image for visualization
        output_image_np = output_image_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
        output_image_np = np.clip(output_image_np.astype(np.uint8), 0, 255)

        # Display the recolored image
        st.image(output_image_np, caption='Recolored Image', use_column_width=True)


