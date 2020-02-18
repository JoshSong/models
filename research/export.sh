INPUT_TYPE=image_tensor
#PIPELINE_CONFIG_PATH=/home/josh/tf-models-repo/research/object_detection/samples/configs/anno_ssd_resnet101_v1_fpn.config
#TRAINED_CKPT_PREFIX=/media/storage/models/anno5/model.ckpt-400000
#EXPORT_DIR=/media/storage/models/anno5/export
PIPELINE_CONFIG_PATH=/home/josh/tf-models-repo/research/object_detection/samples/configs/anno_faster_rcnn_resnet50.config
TRAINED_CKPT_PREFIX=/media/storage/models/anno10/model.ckpt-10000
EXPORT_DIR=/media/storage/models/anno10/export
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}