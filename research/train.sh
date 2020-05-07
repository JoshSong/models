# From the tensorflow/models/research/ directory
#PIPELINE_CONFIG_PATH=/home/josh/tf-models-repo/research/object_detection/samples/configs/anno_faster_rcnn_inception_resnet_v2_atrous.config
PIPELINE_CONFIG_PATH=/home/josh/tf-models-repo/research/object_detection/samples/configs/anno_faster_rcnn_resnet50.config
#PIPELINE_CONFIG_PATH=/home/josh/tf-models-repo/research/object_detection/samples/configs/anno_ssd_resnet101_v1_fpn.config
MODEL_DIR=/media/storage/models/anno19
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
