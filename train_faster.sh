#!/bin/bash 
python3 -u train.py \
    --version $2 \
    --model $1 \
    --checkpoints_dir ./checkpoints \
    --labels_dir ./data/all_labels \
    --images_dir ./data/complete_dataset \
    --split_path ./data/splits/split3.json \
    --dataset AI4EU \
    --wgisd_labels_dir ./data/wgisd \
    --wgisd_images_dir ./data/wgisd \
    --wgisd_split_path ./data/splits_wgisd/split1.json \
    --min_area 0 \
    --annotators 0,4,5,6 \
    --annotator_draw 0 \
    --nepochs 200 \
    --display_freq 250 \
    --save_latest_freq 250 \
    --lr 0.001 \
    --lr_scheduler \
    --batch_size 2 \
    --step_batch_size 1 \
    --box_nms_thresh 1 \
    --transform_min_size 2000 \
    --transform_max_size 2500 \
    --validate_train_split \
    # --partially_pretrained
    #--pretrained
     --custom_anchor_widths \
     --backbone_return_layers 1,2,3 \
