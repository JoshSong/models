#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
source /home/Staff/uqjsong2/tensorflow/bin/activate
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path=/home/Staff/uqjsong2/tfresearch/research/object_detection/samples/configs/ssd_mobilenet_v1_flickr47.config \
    --trained_checkpoint_prefix "/home/Staff/uqjsong2/train_dir3/model.ckpt-237320" \
    --output_directory /home/Staff/uqjsong2/export_dir3
