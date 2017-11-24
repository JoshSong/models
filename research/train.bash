#!/bin/bash
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/home/josh/models/research/object_detection/samples/configs/frcnn_inception_v2_flickr47.config \
    --train_dir=/mnt/disks/disk-1/train_dir
