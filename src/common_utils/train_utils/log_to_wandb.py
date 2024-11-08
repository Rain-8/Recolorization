import wandb
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import lab2rgb



def log_images_and_metrics(src_image, outputs, tgt_palette, step, num_samples=4):
    """
    Log images and metrics to Weights & Biases
    
    Args:
        src_image (torch.Tensor): Source images batch
        outputs (torch.Tensor): Generated images batch
        tgt_palette (torch.Tensor): Target color palettes
        val_loss (float): Validation loss value
        step (int): Current step/iteration
        num_samples (int): Number of samples to visualize
    """
    # Ensure we don't try to log more images than we have in the batch
    num_samples = min(num_samples, src_image.size(0))
    
    # Convert tensors to numpy arrays and move to CPU if needed
    src_imgs = src_image[:num_samples].detach().cpu()
    gen_imgs = outputs[:num_samples].detach().cpu()
    palettes = tgt_palette[:num_samples].detach().cpu()
    
    # Create a figure for palettes
    plt.figure(figsize=(2, num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i + 1)
        palette = palettes[i].reshape(-1, 3)  # Reshape to (num_colors, 3)
        plt.imshow(palette.unsqueeze(1).numpy())  # Add width dimension
        plt.axis('off')
    palette_fig = plt.gcf()
    plt.close()

    # Create grid of images
    src_grid = vutils.make_grid(src_imgs, nrow=num_samples, normalize=True)
    gen_grid = vutils.make_grid(gen_imgs, nrow=num_samples, normalize=True)

    # Log images and metrics to wandb
    wandb.log({
        "validation/source_images": wandb.Image(src_grid.numpy().transpose(1, 2, 0)),
        "validation/generated_images": wandb.Image(gen_grid.numpy().transpose(1, 2, 0)),
        "validation/target_palettes": wandb.Image(palette_fig),
    }, step=step)


def tensor_to_image(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.dim() == 3:
            return tensor.permute(1, 2, 0).numpy()
        return tensor.numpy()
    return tensor


def log_images_and_metrics_custom(src_image, outputs, tgt_palette, step, num_samples=4):
    """
    Log images and metrics to Weights & Biases.
    
    Args:
        src_image (torch.Tensor): Source images batch in LAB format
        outputs (torch.Tensor): Generated images batch in LAB format
        tgt_palette (torch.Tensor): Target color palettes in LAB format
        step (int): Current step/iteration
        num_samples (int): Number of samples to visualize
    """
    # Ensure we don't try to log more images than we have in the batch
    num_samples = min(num_samples, src_image.size(0))
    
    # Convert tensors to numpy arrays and move to CPU if needed
    src_imgs = src_image[:num_samples].detach().cpu()
    gen_imgs = outputs[:num_samples].detach().cpu()
    palettes = tgt_palette[:num_samples].detach().cpu()
    
    src_image_labs = [np.clip(tensor_to_image(img), 0, 100) for img in src_imgs]
    gen_image_labs = [np.clip(tensor_to_image(img), 0, 100) for img in gen_imgs]

    # Convert LAB images to RGB for logging
    src_imgs_rgb = [np.clip(lab2rgb(img)*255, 0, 1) for img in src_image_labs]
    gen_imgs_rgb = [np.clip(lab2rgb(img)*255, 0, 1) for img in gen_image_labs]

    # Convert LAB palettes to RGB for logging
    palettes_rgb = []
    for i in range(num_samples):
        tgt_palette_lab = tensor_to_image(palettes[i])
        palette_rgb = np.clip(lab2rgb(tgt_palette_lab) * 255, 0, 1)
        palettes_rgb.append(palette_rgb.astype(np.uint8))

    # Create wandb images
    wandb_images = {
        "validation/source_images": [wandb.Image(img, caption=f"Source Image {i}") for i, img in enumerate(src_imgs_rgb)],
        "validation/generated_images": [wandb.Image(img, caption=f"Generated Image {i}") for i, img in enumerate(gen_imgs_rgb)],
        "validation/target_palettes": [wandb.Image(palette, caption=f"Target Palette {i}") for i, palette in enumerate(palettes_rgb)],
    }
    
    # Log to wandb
    wandb.log(wandb_images, step=step)