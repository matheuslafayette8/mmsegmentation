#!/bin/bash

MMSEGMENTATION_DIR="/home/openmmlab/mmsegmentation"

cd $MMSEGMENTATION_DIR
python dataset/generate_masks.py
python dataset/generate_mmseg_train_data.py
