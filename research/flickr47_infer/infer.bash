#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
source /home/Staff/uqjsong2/tensorflow/bin/activate
start=`date +%s`
#TF_RECORD_FILES=/home/Staff/uqjsong2/data/test/3ds_output_2_test.record
TF_RECORD_FILES=/home/Staff/uqjsong2/data/test/flickrlogos_47_test.record
PYTHONPATH=$PYTHONPATH:$(readlink -f ..) \
python -m object_detection.inference.infer_detections \
  --input_tfrecord_paths=$TF_RECORD_FILES \
  --output_tfrecord_path=detections.tfrecord \
  --discard_image_pixels \
  --inference_graph=/home/Staff/uqjsong2/export_dir3/frozen_inference_graph.pb
  #--inference_graph=/mnt/disks/disk-1/frcnn_resnet101_flickr47/frozen_inference_graph.pb
  #--inference_graph=/home/josh/davis/models/ssd_mobilenet_v1_flickr47/frozen_inference_graph.pb
  #--inference_graph=/mnt/disks/disk-1/frcnn_inception_v2_flickr47/frozen_inference_graph.pb
end=`date +%s`
runtime=$((end-start))
echo "Done. Took ${runtime} seconds"

echo "Running metrics"

mkdir -p eval_metrics

echo "
label_map_path: '../object_detection/data/flickr47_label_map.pbtxt'
tf_record_input_reader: { input_path: 'detections.tfrecord' }
" > eval_metrics/input_config.pbtxt

echo "
metrics_set: 'open_images_metrics'
" > eval_metrics/eval_config.pbtxt

PYTHONPATH=$PYTHONPATH:$(readlink -f ..) \
python -m object_detection.metrics.offline_eval_map_corloc \
  --eval_dir=eval_metrics \
  --eval_config_path=eval_metrics/eval_config.pbtxt \
  --input_config_path=eval_metrics/input_config.pbtxt
