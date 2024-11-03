import os
import json
from colorthief import ColorThief
from PIL import Image
import random
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2hsv, hsv2rgb
import glob

def extract_top_colors(image_path):
    """Extracts the top colors from the image as RGB tuples."""
    num_colors = random.randint(3, 15)  # randomly setting the length of the palette
    color_thief = ColorThief(image_path)
    return color_thief.get_palette(color_count=num_colors)

def hue_shift_reinforced_palette(palette, hue_shift):
    """Apply a hue shift with luminance preservation to the palette colors."""
    new_palette = []
    for color in palette:
        lab_color = rgb2lab(np.array(color) / 255)
        L_channel = lab_color[0]

        hsv_color = rgb2hsv(np.array(color) / 255)
        hsv_color[0] = (hsv_color[0] + hue_shift / 360) % 1

        hue_shifted_rgb = hsv2rgb(hsv_color) * 255
        hue_shifted_rgb = np.clip(hue_shifted_rgb, 0, 255).astype(np.uint8)

        hue_shifted_lab = rgb2lab(hue_shifted_rgb / 255.0)
        hue_shifted_lab[0] = L_channel

        final_rgb = lab2rgb(hue_shifted_lab) * 255
        final_rgb = np.clip(final_rgb, 0, 255).astype(np.uint8)
        new_palette.append(tuple(final_rgb))

    return new_palette

def recolorize_image_with_palette(image_path, original_palette, new_palette, threshold=30):
    image = Image.open(image_path)
    pixels = np.array(image)

    original_palette_lab = np.array([rgb2lab(np.array(color).reshape(1, 1, 3) / 255.0).flatten() for color in original_palette])
    new_palette_rgb = np.array(new_palette)

    pixels_lab = rgb2lab(pixels / 255.0)

    distances = np.sqrt(np.sum((pixels_lab[:, :, None, :] - original_palette_lab[None, None, :, :]) ** 2, axis=-1))

    closest_color_idx = np.argmin(distances, axis=-1)

    min_distances = np.min(distances, axis=-1)
    within_threshold = min_distances < threshold

    recolored_pixels = pixels.copy()
    recolored_pixels[within_threshold] = new_palette_rgb[closest_color_idx[within_threshold]]

    recolored_image = Image.fromarray(recolored_pixels.astype(np.uint8))
    return recolored_image

def process_images(src_dir, tgt_dir):
    os.makedirs(tgt_dir, exist_ok=True)
    os.makedirs(f"{tgt_dir}/src_images", exist_ok=True)
    os.makedirs(f"{tgt_dir}/tgt_images", exist_ok=True)
    json_data = {}
    image_paths = sorted(glob.glob(f"{src_dir}/*.png"))

    for image_path in image_paths[:10]:
        image_name = os.path.basename(image_path).split(".")[0]
        original_image_path = os.path.join(tgt_dir, f"src_images/{image_name}.png")

        print(f"Processing {image_name}...")

        # Copy original image to target directory
        Image.open(image_path).save(original_image_path)

        # Step 1: Extract top colors from the image
        top_colors = extract_top_colors(image_path)
        hue_shifts = random.sample(range(30, 360), k=3)
        # Process each hue shift
        for hue_shift in hue_shifts:
            hue_shifted_colors = hue_shift_reinforced_palette(top_colors, hue_shift=hue_shift)
            recolored_image_name = f"tgt_images/recolor_{image_name}_hue_shift_{hue_shift}.png"
            recolored_image_path = os.path.join(tgt_dir, recolored_image_name)

            # Step 3: Recolorize the image using the new palette
            recolorized_image = recolorize_image_with_palette(image_path, top_colors, hue_shifted_colors)
            recolorized_image.save(recolored_image_path)

            # Form JSON structure
            json_key = f"{image_name}_hue_shift_{hue_shift}"
            json_data[json_key] = {
                "src_image_path": original_image_path,
                "src_palette": [[int(c) for c in color] for color in top_colors],
                "tgt_palette": [[int(c) for c in color] for color in hue_shifted_colors],
                "tgt_image_path": recolored_image_path
            }

            print(f"Saved recolorized image with hue shift {hue_shift} to {recolored_image_path}")

    # Save JSON to file
    json_output_path = os.path.join(tgt_dir, "recolorization_data.json")
    with open(json_output_path, "w") as json_file:
        json.dump(json_data, json_file)
    print(f"JSON data saved to {json_output_path}")

# Main script
if __name__ == "__main__":
    src_dir = "../../datasets/IMAGE_DATASET_FOR_COLORTHIEF/"  # Directory containing source images
    tgt_dir = "../../datasets/processed_data_v1/"  # Directory to save recolorized images and palettes

    process_images(src_dir, tgt_dir)
