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
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    data = RecolorizeDataset(json_path=dataset_path, transform=transform, sample=sample)

    # Use provided output folder or set a default
    if output_folder is None:
        output_folder = "./converted_images/"

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through the dataset and save each image
    for idx, (image, label) in enumerate(data):
        # Convert tensor to PIL image
        pil_image = transforms.ToPILImage()(image)

        # Define the filename and save the image
        filename = os.path.join(output_folder, f"image_{idx}.png")
        pil_image.save(filename)
        print(f"Saved {filename}")

    return data


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
