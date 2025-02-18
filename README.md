## To Do
- [x] Upload Code.
- [ ] Upload Checkpoint.

## Installation
```shell
cd PixCon
conda env create -f environment.yml
conda activate pixcon
```

## Pre-training on COCO

**Step 0.** Download [COCO dataset](http://images.cocodataset.org/zips/train2017.zip). You can also download the [unlabeled set](http://images.cocodataset.org/zips/unlabeled2017.zip), which combined with the previous downloaded set becomes COCO+.

**Step 1.** Configure the data path.
In configs/selfsup/_base_/datasets/coco_coord.py, modify the data path to COCO
```python
data = dict(
    train=dict(
        main_dir='/data/path/to/coco',
        coco_plus_dir="/data/path/to/unlabeled_set"
    ))
```

**Step 2.** Run pre-training script.
```shell
./scripts/pixcon_sr_resnet50_coco_800ep.sh
```
The pre-training is done on 4 GPUs by default.

## Preparing for Evaluation
**Step 0.** Prepare datasets for detection.

* Assuming that COCO train/val sets have been downloaded already, download the [annonations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip). 

* Download [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and [2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) train sets as well as [VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) test set. 

* Create symbolic links.
```bash
ln -s /data/path/to/coco tools/benchmarks/detectron2/datasets/coco
ln -s /data/path/to/VOC2007 tools/benchmarks/detectron2/datasets/VOC2007
ln -s /data/path/to/VOC2012 tools/benchmarks/detectron2/datasets/VOC2012
```

**Step 1.** Prepare datasets for segmentation.

We follow mmsegmentation's [instruction](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) for dataset preparation. But for it to work under the mmselfsup framework, we need a symbolic link to the datasets

```shell
mkdir data && cd data
ln -s /data/path/to/seg_dataset ${DATASET_NAME}
```

**Step 2.** Extract backbone weights.

```shell
python tools/extract_backbone_weights.py \
work_dirs/selfsup/${EXP_NAME}/epoch_800.pth \
work_dirs/selfsup/${EXP_NAME}/backbone.pth
```

**Step 3.** Convert backbone to detectron2 format.

```shell
python tools/benchmarks/detectron2/convert-pretrain-to-detectron2.py \
work_dirs/selfsup/${EXP_NAME}/backbone.pth \
work_dirs/selfsup/${EXP_NAME}/detectron2.pkl
```

## Object Detection 

We use [detectron2](https://github.com/facebookresearch/detectron2) for fine-tuning the pre-trained models on VOC detection and COCO detection tasks. All the evaluations are done on 4 gpus.

**VOC Detection**
```bash
bash tools/benchmarks/detectron2/run.sh \
configs/benchmarks/detectron2/pascal_voc_R_50_C4_24k_moco.yaml \
work_dirs/selfsup/${EXP_NAME}/detection.pkl \
work_dirs/selfsup/benchmarks/detectron2/voc12/${EXP_NAME}
```

**COCO Detection**
```bash
bash tools/benchmarks/detectron2/run.sh \
configs/benchmarks/detectron2/coco_R_50_FPN_1x_moco.yaml \
work_dirs/selfsup/${EXP_NAME}/detection.pkl \
work_dirs/selfsup/benchmarks/detectron2/coco/${EXP_NAME}
```

## Semantic Segmentation
We use [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) for fine-tuning pre-trained models on semantic segmentations tasks. All the fine-tunings have been conducted on 4 GPUS with a total batch size of 16.

**VOC (aug) Segmentation**

```bash
bash tools/benchmarks/mmsegmentation/mim_dist_train.sh \
configs/benchmarks/mmsegmentation/voc12aug/fcn_d6_r50-d16_513x513_30k_voc12aug_moco.py \
work_dirs/selfsup/${EXP_NAME}/backbone.pth \
4 \
--work-dir work_dirs/selfsup/benchmarks/mmseg/voc12aug/${EXP_NAME}
```

**Cityscapes Segmentation**
```bash
bash tools/benchmarks/mmsegmentation/mim_dist_train.sh \
configs/benchmarks/mmsegmentation/cityscapes/fcn_d6_r50-d16_769x769_90k_cityscapes_moco.py \
work_dirs/selfsup/${EXP_NAME}/backbone.pth \
4 \
--work-dir work_dirs/selfsup/benchmarks/mmseg/cityscapes/${EXP_NAME}

```

