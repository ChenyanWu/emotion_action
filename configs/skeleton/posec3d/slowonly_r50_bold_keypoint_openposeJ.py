model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        in_channels=18,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=26,
        spatial_type='avg',
        dropout_ratio=0.5,
        loss_cls=dict(type='BCELossWithLogits', loss_weight=1.0),
        multi_class=True,
        label_smooth_eps=0,
        ),
    train_cfg=dict(),
    test_cfg=dict(average_clips='prob'))

dataset_type = 'PoseDataset'
ann_file_train = 'data/BOLD_public/annotations/bold.pkl'
ann_file_val = 'data/BOLD_public/annotations/bold.pkl'
left_kp = [5, 6, 7, 11, 12, 13, 15, 17]
right_kp = [2, 3, 4, 8, 9, 10, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=True,
        left_kp=left_kp,
        right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        split='train',
        data_prefix='',
        multi_class=True,
        num_classes=26,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        split='val',
        multi_class=True,
        num_classes=26,
        data_prefix='',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        split='val',
        data_prefix='',
        multi_class=True,
        num_classes=26,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.2, momentum=0.9,
    weight_decay=0.0003)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
# lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
# total_epochs = 240
lr_config = dict(policy='step', step=[90, 110])
total_epochs = 40
checkpoint_config = dict(interval=10)
workflow = [('train', 10)]
evaluation = dict(interval=1, metrics=['mean_average_precision', 'multi_class_AUC'])
# evaluation = dict(
#     interval=10,
#     metrics=['top_k_accuracy', 'mean_class_accuracy'],
#     topk=(1, 5))
log_config = dict(
    interval=20, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/posec3d/bold_openposeJ'
load_from = None
resume_from = None
find_unused_parameters = False
