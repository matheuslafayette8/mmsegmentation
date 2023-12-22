_base_ = [
    '../_base_/models/lane_vec_twins_pcpvt-s_fpn.py', '../_base_/datasets/coco_lane_vec.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (320, 320)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=None)
