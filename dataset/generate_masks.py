from pycocotools.coco import COCO
import cv2
import numpy as np
import os

def get_masks_from_coco_annotation(annotation_file, output_dir):
    # Load COCO annotations
    coco = COCO(annotation_file)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all images in the dataset
    for img_id in coco.getImgIds():
        # Load information for the current image
        img_info = coco.loadImgs(img_id)[0]

        # Load annotations for the image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        # Create an empty mask
        height, width = img_info['height'], img_info['width']
        mask = np.zeros((height, width), dtype=np.uint8)

        # Combine masks for each annotation
        for ann in annotations:
            seg = ann['segmentation']
            cat_id = ann['category_id']
            category_info = coco.loadCats(cat_id)[0]

            # Create binary mask
            binary_mask = coco.annToMask(ann)

            # Combine masks (e.g., for overlapping annotations)
            mask = np.maximum(mask, binary_mask)

        # Save the mask as an image file with the same name as the original image
        image_filename = os.path.basename(img_info['file_name'])
        mask_filename = os.path.join(output_dir, f'{image_filename}')
        cv2.imwrite(mask_filename, mask)

if __name__ == "__main__":
    dataset = '18_12_first_dataset'
    annotation_file = dataset + '/result.json'  # Update with your COCO annotations file
    output_directory = dataset + '/masks/'  # Update with the desired output directory
    
    annotation_file = "dataset/" + annotation_file
    output_directory = "dataset/" + output_directory

    get_masks_from_coco_annotation(annotation_file, output_directory)