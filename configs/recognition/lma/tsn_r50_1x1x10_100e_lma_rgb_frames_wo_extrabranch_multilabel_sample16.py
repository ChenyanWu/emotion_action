_base_ = [
    '../../_base_/schedules/sgd_100e.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=11,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        # use for the multi class
        # loss_cls=dict(type='BCELossWithLogits', loss_weight=1.0),
        loss_cls=dict(type='BCELossWithLogits', loss_weight=1.0, pos_weight=[9 for _ in range(11)]),
        multi_class=True,
        label_smooth_eps=0
        ),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips=None))

# dataset settings
dataset_type = 'LmaframeMultiLabelDataset'
data_root = 'data/BOLD_public/mmextract'
data_root_val = 'data/BOLD_public/mmextract'
ann_file_train = 'data/BOLD_public/annotations/LMA_coding_cleaned_enlarge_train.csv'
ann_file_val = 'data/BOLD_public/annotations/LMA_coding_cleaned_enlarge_val.csv'
ann_file_test = 'data/BOLD_public/annotations/LMA_coding_cleaned_enlarge_val.csv'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

set_clip_num = 16
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
    dict(type='RandomResizedCrop'),
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
    dict(type='CenterCrop', crop_size=256),
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
        num_clips=25,
        test_mode=True),
    dict(type='RawFrameCropDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
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
        multi_class=True,
        num_classes=11,
        ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        lma_annot_idx=set_lma_annot_idx,
        multi_class=True,
        num_classes=11,
        ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        lma_annot_idx=set_lma_annot_idx,
        multi_class=True,
        num_classes=11,
        ))
evaluation = dict(interval=1, metrics=['mean_average_precision', 'multi_class_AUC'])

optimizer = dict(
    type='SGD',
    lr=0.001, # this lr is used for 1 gpus
    # lr=0.00125,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001)

# runtime settings
checkpoint_config = dict(interval=10)
work_dir = './work_dirs/lma_predict/res50_tsn_lma_rgb_multi_class_posweight_frame16/'
# load_from = ('./model_zoo_dirs/resnext101_32x4d_b32x8_imagenet_20210506-e0fa3dd5.pth')
load_from = ('./model_zoo_dirs/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth')
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 50