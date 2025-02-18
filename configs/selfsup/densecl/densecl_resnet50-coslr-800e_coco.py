_base_ = [
    '../_base_/datasets/coco_coord.py',
    '../_base_/default_runtime.py',
]
# model
# model settings
model = dict(
    type='DenseCL',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    loss_lambda=0.5,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='DenseCLNeck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        num_grid=None,
        syncbn=False),
    predictor=dict(
        type='MoCoV2Neck',
        in_channels=128,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=False,
        syncbn=False), 
    head=dict(type='ContrastiveHead', temperature=0.2))
# data
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    drop_last=True,
    train=dict(
        return_coords=False)
    )
two_crop = dict(type="CustomTwoCrop", size=224, scale=(0.2, 1.0), ratio=(3. / 4., 4. / 3.))
hflip = dict(type="CustomRandomHorizontalFlip", p=0.5)
train_pipeline1 = [
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
    dict(type='ToGray', p=0.2),
    dict(type='GaussianBlur', blur_limit=(23, 23), sigma_limit=(0.1, 2.0), p=0.5), 
    dict(type='Normalize', mean= (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0), 
    dict(type='ToTensorV2'), 
]
train_pipeline1 = [
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
    dict(type='ToGray', p=0.2),
    dict(type='GaussianBlur', blur_limit=(23, 23), sigma_limit=(0.1, 2.0), p=0.5), 
    dict(type='Normalize', mean= (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0), 
    dict(type='ToTensorV2'), 
]
# optimizer
update_interval=1
optimizer = dict(type='SGD', lr=0.3, weight_decay=0.0001, momentum=0.9)
optimizer_config = dict(update_interval=update_interval)
# fp16
fp16 = dict(loss_scale='dynamic')
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=4,
    warmup_ratio=0.0001, # cannot be 0
    warmup_by_epoch=True)
checkpoint_config = dict(interval=50)

# runtime settings
runner = dict(type='EpochBasedRunnerProgressBar', max_epochs=800)
find_unused_parameters = False