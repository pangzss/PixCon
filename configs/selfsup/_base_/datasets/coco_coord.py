# dataset settings
dataset_type = 'COCOCoord'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

two_crop = dict(type="CustomTwoCrop", size=224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
hflip = dict(type="CustomRandomHorizontalFlip", p=0.5)
train_pipeline1 = [
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
    dict(type='ToGray', p=0.2),
    dict(type='GaussianBlur', blur_limit=(23, 23), sigma_limit=(0.1, 2.0), p=1.0), 
    dict(type='Solarize', threshold=128, p=0.0), 
    dict(type='Normalize', mean= (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0), 
    dict(type='ToTensorV2'), 
]
train_pipeline2 = [
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
    dict(type='ToGray', p=0.2),
    dict(type='GaussianBlur', blur_limit=(23, 23), sigma_limit=(0.1, 2.0), p=0.1), 
    dict(type='Solarize', threshold=128, p=0.2), 
    dict(type='Normalize', mean= (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0), 
    dict(type='ToTensorV2'), 
]

# dataset summary
data = dict(
    samples_per_gpu=128, 
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        dataset="COCO",
        main_dir='/data/path/to/coco',
        split='train2017',
        two_crop=two_crop,
        hflip=hflip,
        pipelines=[train_pipeline1, train_pipeline2]
    ))
