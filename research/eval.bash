#!/bin/bash
python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=/home/josh/models/research/object_detection/samples/configs/frcnn_inception_v2_flickr47.config \
    --checkpoint_dir=/mnt/disks/disk-1/train_dir \
    --eval_dir=/mnt/disks/disk-1/eval_dir
