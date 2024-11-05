import os
import shutil
import random

def split_dataset(src_folder, dest_folder, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05):
    """
    Splits images from the src_folder into train, val, and test folders inside dest_folder
    with the specified ratios.

    :param src_folder: Path to the source folder containing images.
    :param dest_folder: Path to the destination folder where train, val, and test folders will be created.
    :param train_ratio: Proportion of images to go into the training set.
    :param val_ratio: Proportion of images to go into the validation set.
    :param test_ratio: Proportion of images to go into the test set.
    """
    # Verify that the ratios sum to 1.0
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    
    # Create destination subfolders
    train_folder = os.path.join(dest_folder, "train")
    val_folder = os.path.join(dest_folder, "val")
    test_folder = os.path.join(dest_folder, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # Get all image files from the source folder
    images = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    random.shuffle(images)  # Shuffle images for random distribution
    
    # Calculate split indices
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    
    # Split the images
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]
    
    # Move images to the respective folders
    for image in train_images:
        shutil.copy(os.path.join(src_folder, image), os.path.join(train_folder, image))
        
    for image in val_images:
        shutil.copy(os.path.join(src_folder, image), os.path.join(val_folder, image))
        
    for image in test_images:
        shutil.copy(os.path.join(src_folder, image), os.path.join(test_folder, image))
    
    print(f"Total images: {total_images}")
    print(f"Train set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    print(f"Test set: {len(test_images)} images")

# Example usage

if __name__=="__main__":
    src_folder = "../../datasets/design_seed_by_labels"  # Folder containing all images
    dest_folder = "../../datasets/raw_recolor_data_sample"  # Folder where train, val, test folders will be created

    split_dataset(src_folder, dest_folder)
