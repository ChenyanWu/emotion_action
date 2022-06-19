_base_ = [
    '../../_base_/schedules/sgd_50e.py', '../../_base_/default_runtime.py'
]

set_clip_len = 5
# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet101',
        in_channels= 2*set_clip_len, # 'in_channels' should be 2 * clip_len
        depth=101,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=26,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        loss_cls=dict(type='BCELossWithLogits', loss_weight=1.0),
        dropout_ratio=0.5,
        init_std=0.01,
        multi_class=True,
        label_smooth_eps=0),
    train_cfg=None,
    # train_cfg=dict(aux_info='meta_label'),
    test_cfg=dict(average_clips=None))

# dataset settings
dataset_type = 'BoldframeDataset'
data_root = 'data/BOLD_public/mmflow'
data_root_val = 'data/BOLD_public/mmflow'
# ann_file_train = 'data/BOLD_public/annotations/train_frame_list.txt'
# ann_file_val = 'data/BOLD_public/annotations/val_frame_list.txt'
# ann_file_test = 'data/BOLD_public/annotations/val_frame_list.txt'
ann_file_train = 'data/BOLD_public/annotations/train.csv'
ann_file_val = 'data/BOLD_public/annotations/val.csv'
ann_file_test = 'data/BOLD_public/annotations/bold_test_ijcv.csv'
img_norm_cfg = dict(mean=[128, 128], std=[128, 128])
train_pipeline = [
    dict(type='SampleFrames', clip_len=set_clip_len, frame_interval=1, num_clips=10),
    dict(type='RawFrameDecode'),
    # dict(type='RawFrameCropDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW_Flow'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=set_clip_len,
        frame_interval=1,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    # dict(type='RawFrameCropDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW_Flow'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=set_clip_len,
        frame_interval=1,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    # dict(type='RawFrameCropDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW_Flow'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='flow_{}_{:05d}.jpg',
        modality='Flow',
        pipeline=train_pipeline,
        multi_class=True,
        num_classes=26),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='flow_{}_{:05d}.jpg',
        modality='Flow',
        pipeline=val_pipeline,
        multi_class=True,
        num_classes=26),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='flow_{}_{:05d}.jpg',
        modality='Flow',
        pipeline=test_pipeline,
        multi_class=True,
        num_classes=26))
evaluation = dict(interval=1, metrics=['mean_average_precision'])

optimizer = dict(
    type='SGD',
    lr=0.005,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001)
lr_config = dict(policy='step', step=[70])

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/tsn_r101_1x1x5_50e_bold_rgb_flow_lr/'
