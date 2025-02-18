#!/bin/bash

exp="pixcon_sr_resnet50-coslr-800e_coco"

bash tools/dist_train.sh \
configs/selfsup/pixcon/${exp}.py \
4 \
--work_dir work_dirs/selfsup/${exp}