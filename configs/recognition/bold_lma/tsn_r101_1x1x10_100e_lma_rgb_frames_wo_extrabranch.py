_base_ = [
    '../../_base_/schedules/sgd_100e.py',
    '../../_base_/default_runtime.py'
]
# _base_ = [
#     '../../_base_/models/tsn_r50.py', '../../_base_/schedules/sgd_100e.py',
#     '../../_base_/default_runtime.py'
# ]

# model settings
# model = dict(cls_head=dict(num_classes=5))
# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        # type='ResNetMultiHead',
        type='ResNet',
        pretrained='torchvision://resnet101',
        depth=101,
        norm_eval=False),
    cls_head=dict(
        # type='TSNMultiHead',
        type='TSNHead',
        num_classes=2,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        # dropout_ratio=0.5,
        dropout_ratio=0.3,
        init_std=0.01,
        # use for the multi class
        # loss_cls=dict(type='BCELossWithLogits', loss_weight=1.0),
        # multi_class=True,
        # label_smooth_eps=0
        # use for the single class
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[1.,3.0]),
        ),
    train_cfg=None,
    # train_cfg=dict(aux_info=['meta_label']),
    test_cfg=dict(average_clips=None))

# dataset settings
dataset_type = 'LmaframeDataset'
data_root = 'data/BOLD_public/mmextract'
data_root_val = 'data/BOLD_public/mmextract'
ann_file_train = 'data/BOLD_public/annotations/LMA_coding_cleaned_train.csv'
ann_file_val = 'data/BOLD_public/annotations/LMA_coding_cleaned_val.csv'
ann_file_test = 'data/BOLD_public/annotations/LMA_coding_cleaned_val.csv'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

set_clip_num = 20
# set_clip_num = 40
set_clip_len = 1
# set_clip_len = 3
set_frame_interval = 1
# set_frame_interval = 3

train_pipeline = [
    dict(type='SampleFrames', clip_len=set_clip_len, frame_interval=set_frame_interval, num_clips=set_clip_num),
    dict(type='RawFrameCropDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label', 'meta_label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'meta_label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=set_clip_len,
        frame_interval=set_frame_interval,
        num_clips=set_clip_num,
        test_mode=True),
    dict(type='RawFrameCropDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=set_clip_len,
        frame_interval=set_frame_interval,
        num_clips=set_clip_num,
        test_mode=True),
    dict(type='RawFrameCropDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

optimizer = dict(
    type='SGD',
    lr=0.001, # this lr is used for 1 gpus
    # lr=0.00125,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001)

# runtime settings
work_dir = './work_dirs/lma_rgb_tsn_r101_frame_1x1x10_100e/'
