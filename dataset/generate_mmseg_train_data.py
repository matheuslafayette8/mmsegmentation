import os
import shutil

def copy_images(src_folder, dest_folder, split):
    src_path = os.path.join(src_folder, "images")
    dest_path = os.path.join(dest_folder, "images", split + "2017")

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    for filename in os.listdir(src_path):
        if filename.endswith(".png"):
            img_path = os.path.join(src_path, filename)
            dest_filename = os.path.join(dest_path, os.path.splitext(filename)[0] + ".jpg")
            shutil.copy(img_path, dest_filename)

def copy_masks(src_folder, dest_folder, split):
    src_path = os.path.join(src_folder, "masks")
    dest_path = os.path.join(dest_folder, "annotations", split + "2017")

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    for filename in os.listdir(src_path):
        if filename.endswith(".png"):
            mask_path = os.path.join(src_path, filename)
            dest_filename = os.path.join(dest_path, os.path.splitext(filename)[0] + "_labelTrainIds.png")
            shutil.copy(mask_path, dest_filename)

def main():
    dataset_folder = "dataset/18_12_first_dataset"
    coco_lane_vec_folder = "data/coco_lane_vec"

    # Train set
    copy_images(dataset_folder, coco_lane_vec_folder, "train")
    copy_masks(dataset_folder, coco_lane_vec_folder, "train")

    # Validation set
    copy_images(dataset_folder, coco_lane_vec_folder, "val")
    copy_masks(dataset_folder, coco_lane_vec_folder, "val")

if __name__ == "__main__":
    main()
