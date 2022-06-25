_base_ = [
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
        type='HRNet',
        in_channels=3,
        pretrained='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        # pretrained='https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        frozen_stages=-1,
        ),
    cls_head=dict(
        type='TSNHead',
        num_classes=2,
        in_channels=512,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        # dropout_ratio=0.5,
        dropout_ratio=0.3,
        # dropout_ratio=0.0,
        init_std=0.01,
        # use for the multi class
        # loss_cls=dict(type='BCELossWithLogits', loss_weight=1.0),
        # multi_class=True,
        # label_smooth_eps=0
        # use for the single class
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[1.,10.0]),
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
set_lma_annot_idx = 4

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
        pipeline=train_pipeline,
        lma_annot_idx=set_lma_annot_idx,
        ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        lma_annot_idx=set_lma_annot_idx,
        ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        lma_annot_idx=set_lma_annot_idx,
        ))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer = dict(
#     type='SGD',
#     lr=0.001, # this lr is used for 1 gpus
#     # lr=0.00125,  # this lr is used for 8 gpus
#     momentum=0.9,
#     weight_decay=0.0001)

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
# optimizer_config = dict(grad_clip=None)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    # warmup='linear',
    # warmup_iters=500,
    # warmup_ratio=0.001,
    step=[60, 80])

# runtime settings
checkpoint_config = dict(interval=10)
work_dir = './work_dirs/hrnet_lma_rgb/'
total_epochs = 100
