_base_ = [
    '../_base_/datasets/coco_coord.py',
    '../_base_/default_runtime.py',
]
# model
# model settings
model = dict(
    type='PixCon',
    queue_len=65536,
    feat_dim=128,
    momentum=0.99,
    loss_lambda=0.5,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='DenseCLNeck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        num_grid=None,
        syncbn=True),
    predictor=dict(
        type='MoCoV2Neck',
        in_channels=128,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=False,
        syncbn=True), 
    head=dict(type='ContrastiveHead', temperature=0.2))

# additional hooks
# interval for accumulate gradient, total 16*256*1(interval)=4096
update_interval = 1
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval), 
]
# data
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    drop_last=True,
    train=dict(
        return_coords=False)
    )
# optimizer
optimizer = dict(type='SGD', lr=0.4, weight_decay=0.0001, momentum=0.9)
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