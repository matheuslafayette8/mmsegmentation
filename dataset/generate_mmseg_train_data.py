import os
import shutil
import random

def shuffle_and_copy(src_folder, dest_folder, percentage, random_seed):
        
    src_images_path = os.path.join(src_folder, "imgs")
    src_masks_path = os.path.join(src_folder, "labels")
    
    dest_images_path = os.path.join(dest_folder, "img_dir")
    dest_masks_path = os.path.join(dest_folder, "ann_dir")
    
    dest_train_images_path = os.path.join(dest_images_path, "train")
    dest_train_masks_path = os.path.join(dest_masks_path, "train")
    
    dest_val_images_path = os.path.join(dest_images_path, "val")
    dest_val_masks_path = os.path.join(dest_masks_path, "val")
    
    dest_paths = [dest_train_images_path, dest_train_masks_path, dest_val_images_path, dest_val_masks_path]
    for dest_path in dest_paths:
        print(dest_path)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

    # List all image files
    image_files = [filename for filename in os.listdir(src_images_path) if filename.endswith(".png")]

    # Shuffle the list
    random.seed(random_seed)
    random.shuffle(image_files)

    # Calculate the split index
    split_index = int(len(image_files) * percentage)

    # Split the list into training and validation sets
    train_images = image_files[:split_index]
    val_images = image_files[split_index:]

    # Copy images for training set
    for filename in train_images:
        img_path = os.path.join(src_images_path, filename)
        dest_filename = os.path.join(dest_train_images_path, filename)
        shutil.copy(img_path, dest_filename)

        # Copy corresponding masks
        mask_path = os.path.join(src_masks_path, filename)
        dest_mask_filename = os.path.join(dest_train_masks_path, filename)
        shutil.copy(mask_path, dest_mask_filename)

    # Copy images for validation set
    for filename in val_images:
        img_path = os.path.join(src_images_path, filename)
        dest_filename = os.path.join(dest_val_images_path, filename)
        shutil.copy(img_path, dest_filename)

        # Copy corresponding masks
        mask_path = os.path.join(src_masks_path, filename)
        dest_mask_filename = os.path.join(dest_val_masks_path, filename)
        shutil.copy(mask_path, dest_mask_filename)

def main():
    dataset_name = "08_05_new_data_removed_labels"
    dataset_folder = "dataset/complete_datasets/" + dataset_name
    coco_lane_vec_folder = "data/bev_dataset"

    # Define the percentage for training and validation
    train_percentage = 0.75
    val_percentage = 1 - train_percentage
    random_seed = 42
    
    shuffle_and_copy(dataset_folder, coco_lane_vec_folder, train_percentage, random_seed)

if __name__ == "__main__":
    main()
