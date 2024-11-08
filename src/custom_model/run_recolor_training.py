import argparse
from data import get_data
from model import get_model
from train_recolor import RecolorizeTrainer
from PIL import Image
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Training arguments for the Recolorization Trainer")

    # Model and dataset parameters
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--sample", type=int, default=None, help="Data samples")

    # Logging and validation intervals
    parser.add_argument("--logging_interval", type=int, default=1, help="Interval (in epochs) for logging training loss to WandB")
    parser.add_argument("--validation_interval", type=int, default=1, help="Interval (in epochs) for running evaluation")
    parser.add_argument("--checkpointing_interval", type=int, default=1, help="Interval (in epochs) for saving checkpoint")

    # WandB parameters
    parser.add_argument("--project_name", type=str, default="Recolorization", help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name for WandB. If None, will be timestamped")

    # Dataset paths
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to the validation data")

    #checkpoint dir
    parser.add_argument("--checkpoint_dir", type=str, default="recolor_model_ckpts", help="Directory to save checkpoints")

    args = parser.parse_args()
    return args
    


def get_data(dataset_path, output_folder=None, sample=None):
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Initialize the dataset
    data = RecolorizeDataset(json_path=dataset_path, transform=transform, sample=sample)

    # Use provided output folder or set a default
    if output_folder is None:
        output_folder = "./saved_images/"

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through the dataset and process each sample
    for idx in range(len(data)):
        src_image_lab, tgt_image_lab, illu, src_palette, tgt_palette = data[idx]

        # Convert source image from LAB to RGB
        src_image_np = src_image_lab.detach().cpu().numpy().transpose(1, 2, 0)  # Shape: (H, W, C)
        src_image_rgb = (lab2rgb(src_image_np) * 255).astype(np.uint8)
        src_image_pil = Image.fromarray(src_image_rgb)

        # Save source image
        src_image_filename = os.path.join(output_folder, f"src_image_{idx}.png")
        src_image_pil.save(src_image_filename)
        print(f"Saved {src_image_filename}")

        # Convert target image from LAB to RGB
        tgt_image_np = tgt_image_lab.detach().cpu().numpy().transpose(1, 2, 0)  # Shape: (H, W, C)
        tgt_image_rgb = (lab2rgb(tgt_image_np) * 255).astype(np.uint8)
        tgt_image_pil = Image.fromarray(tgt_image_rgb)

        # Save target image
        tgt_image_filename = os.path.join(output_folder, f"tgt_image_{idx}.png")
        tgt_image_pil.save(tgt_image_filename)
        print(f"Saved {tgt_image_filename}")

        # Save source palette
        src_palette_filename = os.path.join(output_folder, f"src_palette_{idx}.png")
        save_palette_image(src_palette, src_palette_filename)

        # Save target palette
        tgt_palette_filename = os.path.join(output_folder, f"tgt_palette_{idx}.png")
        save_palette_image(tgt_palette, tgt_palette_filename)

    return data

def save_palette_image(palette_lab, filename):
    """
    Convert and save the palette as a visual grid image.
    """
    num_colors = palette_lab.shape[0]
    palette_image = np.zeros((20, num_colors * 20, 3), dtype=np.uint8)

    for i, color_lab in enumerate(palette_lab):
        # Convert LAB color to RGB
        color_lab = color_lab.reshape(1, 1, 3)
        color_rgb = (lab2rgb(color_lab) * 255).astype(np.uint8).squeeze()

        # Fill the corresponding block in the palette image with the RGB color
        col_start = i * 20
        col_end = col_start + 20
        palette_image[:, col_start:col_end, :] = color_rgb

    # Convert to PIL image and save
    palette_pil = Image.fromarray(palette_image)
    palette_pil.save(filename)
    print(f"Saved palette image: {filename}")

        

    


def run_training():
    args = parse_args()
    model = get_model()
    #train_data = get_data(args.train_data_path)
    #val_data = get_data(args.val_data_path)
    # Convert and save train/validation images before training
    #tensor is in the format (C, H, W) with pixel values normalized to [0, 1]
    train_data = get_data(args.train_data_path, output_folder="./train_images/")
    val_data = get_data(args.val_data_path, output_folder="./val_images/")
    trainer = RecolorizeTrainer(model, train_dataset=train_data, eval_dataset=val_data, args=args)
    trainer.train()

if __name__ == "__main__":
    run_training()
