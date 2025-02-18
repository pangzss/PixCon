_base_ = [
    '../_base_/models/byol.py',
    '../_base_/datasets/coco_byol.py',
    '../_base_/default_runtime.py',
]

# additional hooks
# interval for accumulate gradient, total 16*256*1(interval)=4096
update_interval = 1
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval)
]

# optimizer
optimizer = dict(type='SGD', lr=0.4, weight_decay=0.0001, momentum=0.9)
optimizer_config = dict(update_interval=update_interval)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=4,
    warmup_ratio=0.0001, # cannot be 0
    warmup_by_epoch=True)
checkpoint_config = dict(interval=10)

# runtime settings
runner = dict(type='EpochBasedRunnerProgressBar', max_epochs=800)
resume_from = "work_dirs/selfsup/byol_resnet50_bs512-coslr-800e_coco/latest.pth"