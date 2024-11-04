import wandb
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


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

