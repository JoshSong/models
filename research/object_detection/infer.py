import sys
import os
import cv2
import time
import numpy as np
import tensorflow as tf
import json

file_dir = os.path.dirname(os.path.realpath(__file__))
up_dir = os.path.dirname(file_dir)
sys.path.append(up_dir)
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

detection_graph = None
sess = None
category_index = None

def detect_objects(image_np):
    global detection_graph, sess, category_index
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=5,
        min_score_thresh=0.5)
    return image_np

def init_deep():
    global detection_graph, sess, category_index

    graph_path = 'node_models/robo_v1.pb'
    labels_path = 'data/robo_v1_label_map.pbtxt'

    category_index = label_map_util.create_category_index_from_labelmap(labels_path)

    print 'Loading tf model into memory'
    config = tf.ConfigProto()
    detection_graph = tf.Graph()

    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph, config=config)
    print 'Done'

if __name__ == '__main__':
    init_deep()
    image_path = sys.argv[1]
    img = cv2.imread(image_path)
    vis = detect_objects(img)
    cv2.imwrite('infer_vis.jpg', vis)
    cv2.waitKey()
