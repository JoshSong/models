#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
source /home/Staff/uqjsong2/tensorflow/bin/activate
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/home/Staff/uqjsong2/tfresearch/research/object_detection/samples/configs/ssd_mobilenet_v1_flickr47.config \
    --train_dir=/home/Staff/uqjsong2/train_dir3
