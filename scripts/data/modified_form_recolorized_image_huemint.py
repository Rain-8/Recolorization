import os
import json

from networkx import adjacency_matrix
from colorthief import ColorThief
from PIL import Image
import random
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2hsv, hsv2rgb
import glob
from scipy.spatial import KDTree
from get_contrast_matrix import contrast_matrix_flat_list
from get_huemint_response import get_huemint_response
from Pylette import extract_colors
import colorsys
import tqdm

# Extracts the top colors from an image for use in palette generation
def extract_top_colors(image_path):
    """Extracts the top colors from the image as RGB tuples."""
    num_colors = 6  # Setting the length of the palette for design seed images 
    colors = extract_colors(image_path, num_colors)
    colors = [x.rgb for x in colors]
    return colors

# Purpose: To make the palette more cohesive and avoid abrupt color changes.
def hue_shift_reinforced_palette(palette, hue_shift):
    """Apply a hue shift with luminance preservation to the palette colors."""
    new_palette = []
    for color in palette:
        # Convert RGB to LAB color space to isolate luminance
        lab_color = rgb2lab(np.array(color) / 255)
        L_channel = lab_color[0]  # Preserve luminance
        
        # Apply hue shift in HSV space
        hsv_color = rgb2hsv(np.array(color) / 255)
        hsv_color[0] = (hsv_color[0] + hue_shift / 360) % 1  # Shift hue
        
        # Convert back to RGB
        hue_shifted_rgb = hsv2rgb(hsv_color) * 255
        hue_shifted_rgb = np.clip(hue_shifted_rgb, 0, 255).astype(np.uint8)
        
        # Re-apply preserved luminance from LAB color
        hue_shifted_lab = rgb2lab(hue_shifted_rgb / 255.0)
        hue_shifted_lab[0] = L_channel
        final_rgb = lab2rgb(hue_shifted_lab) * 255
        final_rgb = np.clip(final_rgb, 0, 255).astype(np.uint8)
        new_palette.append(tuple(final_rgb))
    return new_palette

def rgb_to_hsv(color):
    return colorsys.rgb_to_hsv(color[0] / 255, color[1] / 255, color[2] / 255)

def hsv_to_rgb(color):
    rgb = colorsys.hsv_to_rgb(color[0], color[1], color[2])
    return tuple(int(c * 255) for c in rgb)

def adjust_hue(color, hue_shift):
    # Convert RGB to HSV
    hsv = rgb_to_hsv(color)
    # Apply hue shift, keeping hue within the [0, 1] range
    new_hue = (hsv[0] + hue_shift) % 1.0
    # Convert back to RGB
    return hsv_to_rgb((new_hue, hsv[1], hsv[2]))

def blend_colors(color1, color2, weight):
    """Blend two colors with a specified weight (0 to 1)."""
    return (1 - weight) * color1 + weight * color2

# Function to blend colors gradually for smooth transitions
def replace_colors_fast(image_path, src_palette, tgt_palette, distance_threshold=50):
    """
    Quickly replace colors in an image by blending colors based on proximity
    to the source palette using KDTree for fast lookup.
    """
    image = Image.open(image_path).convert("RGB")
    pixels = np.array(image, dtype=np.float32).reshape(-1, 3)  # Flattened array
    
    # Create KDTree from the source palette
    src_palette_tree = KDTree(src_palette)
    
    # Find nearest source palette color for each pixel
    distances, src_indices = src_palette_tree.query(pixels)
    
    src_palette = np.array(src_palette, dtype=np.float32)
    tgt_palette = np.array(tgt_palette, dtype=np.float32)
    recolored_pixels = pixels.copy()
    
    # Apply target color blend based on distance threshold
    within_threshold = distances < distance_threshold
    weights = 1 - (distances[within_threshold] / distance_threshold)
    recolored_pixels[within_threshold] = (
        (1 - weights[:, None]) * pixels[within_threshold] +
        weights[:, None] * tgt_palette[src_indices[within_threshold]]
    )

    # Reshape pixels to original image shape
    recolored_pixels = recolored_pixels.reshape(image.size[1], image.size[0], 3).astype(np.uint8)
    recolored_image = Image.fromarray(recolored_pixels)
    return recolored_image

# Converts hex color to RGB format
def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

# Main function to process and recolor images
def process_images(src_dir, tgt_dir):
    os.makedirs(tgt_dir, exist_ok=True)
    os.makedirs(f"{tgt_dir}/src_images", exist_ok=True)
    os.makedirs(f"{tgt_dir}/tgt_images", exist_ok=True)
    json_data = {}
    image_paths = sorted(glob.glob(f"{src_dir}/*"))

    for image_path in tqdm.tqdm(image_paths):
        try:
            image_name = os.path.basename(image_path).split(".")[0]
            original_image_path = os.path.join(tgt_dir, f"src_images/{image_name}.png")
            Image.open(image_path).convert("RGB").save(original_image_path)

            # Step 1: Extract top colors from the image
            top_colors = extract_top_colors(image_path)

            # NEW: Apply hue shift to create a more uniform and balanced palette
            # Reason: Smooths out color transitions and maintains luminance
            hue_shift = random.uniform(-15, 15)  # Adding slight random shift
            adjusted_palette = hue_shift_reinforced_palette(top_colors, hue_shift)

            # Step 2: Generate contrast matrix from adjusted palette for hue-mint response
            adjacency_matrix = contrast_matrix_flat_list(adjusted_palette)
            response = get_huemint_response(adjacency_matrix, len(adjusted_palette))

            # Ensure response is received and generate recolorized images
            if response is not None:
                for idx, result in enumerate(response["results"][:140]):
                    tgt_palette = [hex_to_rgb(color) for color in result["palette"]]
                    recolored_image_name = f"tgt_images/recolor_{image_name}_palette_{idx}.png"
                    recolored_image_path = os.path.join(tgt_dir, recolored_image_name)
                    
                    try:
                        # Updated: Use adjusted_palette instead of top colors
                        recolorized_image = replace_colors_fast(image_path, adjusted_palette, tgt_palette)
                        recolorized_image.convert("RGB").save(recolored_image_path)
                    except Exception as recolor_error:
                        print("Recoloring failed:", recolor_error)
                        continue

                    # Step 4: Form JSON structure
                    json_key = f"{image_name}_palette_{idx}"
                    json_data[json_key] = {
                        "src_image_path": original_image_path,
                        "src_palette": [[int(c) for c in color] for color in top_colors],
                        "tgt_palette": [[int(c) for c in color] for color in tgt_palette],
                        "tgt_image_path": recolored_image_path
                    }

        except Exception as e:
            print("Error: ", e)

        except Exception as e:
            print("Error: ", e)

    # Save JSON to file
    json_output_path = os.path.join(tgt_dir, "recolorization_data.json")
    with open(json_output_path, "w") as json_file:
        json.dump(json_data, json_file)
    print(f"JSON data saved to {json_output_path}")

# Main script
if __name__ == "__main__":
    src_dir = "../../datasets/raw_recolor_data_sample/"  # Directory containing source images
    tgt_dir = "../../datasets/processed_palettenet_data_sample/"  # Directory to save recolorized images and palettes

    for split in ["train", "val", "test"]:
        process_images(f"{src_dir}{split}", f"{tgt_dir}{split}")

