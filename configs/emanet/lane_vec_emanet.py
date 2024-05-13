_base_ = [
    '../_base_/models/emanet_r50-d8.py', '../_base_/datasets/coco_lane_vec.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (320, 320)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

