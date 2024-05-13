#!/bin/bash

MMSEGMENTATION_DIR="/home/openmmlab/mmsegmentation"
SRC_FOLDER="dataset/new_sub_datasets/roc2_road1/rgb_imgs"
CONFIG="configs/bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024.py"
CHECKPOINT="work_dirs/bisenetv2/iter_5000.pth"
DST_FOLDER="dataset/new_sub_datasets/roc2_road1/output/"
OPACITY=0.6

cd $MMSEGMENTATION_DIR
python demo/images_demo.py $SRC_FOLDER $CONFIG $CHECKPOINT $DST_FOLDER --opacity $OPACITY --with-labels