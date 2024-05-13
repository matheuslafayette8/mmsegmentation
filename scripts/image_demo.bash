#!/bin/bash

MMSEGMENTATION_DIR="/home/openmmlab/mmsegmentation"
INPUT_FILE="data/coco_lane_vec/images/train2017/8101668e-i_270_int_img.jpg"
CONFIG="work_dirs/lane_vec_bisenetv1_r18-d32_4xb4-160k_coco-stuff164k-512x512/lane_vec_bisenetv1_r18-d32_4xb4-160k_coco-stuff164k-512x512.py"
CHECKPOINT="work_dirs/lane_vec_bisenetv1_r18-d32_4xb4-160k_coco-stuff164k-512x512/iter_16000.pth"
OUTPUT_FILE="result.png"
OPACITY=0.4

cd $MMSEGMENTATION_DIR
python demo/image_demo.py $INPUT_FILE $CONFIG $CHECKPOINT --out-file $OUTPUT_FILE --opacity $OPACITY