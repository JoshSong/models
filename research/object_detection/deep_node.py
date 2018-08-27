#!/usr/bin/env python

import sys
import os
import json
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
import time

import tensorflow as tf

file_path = os.path.realpath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(file_path)))
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

width = 480
height = 360
#width = 960
#height = 720
num_scans = 10

saved_count = 0

bridge = CvBridge()
cv_image = None
image_sub = None
image_pub = None
int_pub = None

detection_graph = None
sess = None
category_index = None

execution_time = None

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(FILE_DIR, 'deep_node_out')

def detect_objects(image_np):
    global detection_graph, sess, category_index

    # Crop image
    """
    row_off = (image_np.shape[0]-height)/2
    col_off = (image_np.shape[1]-width)/2
    image_np = image_np[row_off:row_off+height, col_off:col_off+width]
    """

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
        min_score_thresh=0.2,
        line_thickness=5)

    return image_np, scores, classes

def ros_service(req):
    global cv_image, saved_count
    print 'called'

    num_detections = [0, 0]
    avg_scores = [0, 0]
    for i in range(num_scans):

        # Wait for new image
        cv_image = None
        while (cv_image is None):
            rospy.sleep(0.5)

        # Run deep classifier
        output_img, scores, classes = detect_objects(cv_image)

        # Save visualization image
        cv2.imwrite(os.path.join(OUTPUT_DIR, str(saved_count) + '.jpg'), output_img)
        saved_count += 1

        # Just take the top detection
        score = scores[0][0]
        logo = classes[0][0]
        avg_scores[logo - 1] += score
        num_detections[logo - 1] += 1

    for i in range(len(avg_scores)):
        if num_detections[i] > 0:
            avg_scores[i] /= num_detections[i]

    ret = 0
    if max(avg_scores[1], avg_scores[0]) > 0.1:
        if avg_scores[1] > avg_scores[0]:
            ret = 2
        else:
            ret = 1

    return TriggerResponse(success=True, message=str(ret))

def callback(data):
    global cv_image, int_pub, execution_time
    print 'Got msg'
    start = time.time()
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    output_img, scores, classes = detect_objects(cv_image)
    print scores
    print classes
    took = time.time() - start
    if execution_time is None:
        execution_time = took
    else:
        execution_time = 0.2 * took + 0.8 * execution_time
    print 'Avg execution time: {} seconds'.format(execution_time)

    logo = 0
    if scores[0][0] > 0.1:
        logo = classes[0][0]
    logo = 5
    int_pub.publish(logo)

    cv2.imshow("Image window", output_img)
    cv2.waitKey(30)

def init_deep():
    global detection_graph, sess, category_index
    model_config = './node_models/ssd_v2_coco.json'
    model_details = json.load(open(model_config))

    print 'Init deep'

    # Path to frozen detection graph.
    # This is the actual model that is used for the object detection.
    MODEL_NAME = model_details['model_folder']
    MODEL_FILE = model_details['model_filename']
    PATH_TO_CKPT = os.path.join(FILE_DIR, 'node_models', MODEL_NAME, MODEL_FILE)

    # List of the strings that is used to add correct label for each box.
    LABELS_FILE = model_details['labels_filename']
    PATH_TO_LABELS = os.path.join(FILE_DIR, 'node_models', 'labels', LABELS_FILE)

    NUM_CLASSES = model_details['number_of_classes']

    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                max_num_classes=NUM_CLASSES,
                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load a (frozen) Tensorflow model into memory.
    print 'Loading tf model into memory'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # Allows GPU memory usage to grow as needed - Added by BG
    detection_graph = tf.Graph()

    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph, config=config)
    print 'Done'

def main():
    global image_sub, image_pub, int_pub

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    init_deep()

    rospy.init_node('tensorflow')
    s = rospy.Service('/tensorflow_srv', Trigger, ros_service)

    int_pub = rospy.Publisher("/tensorflow_msg", Int32, queue_size=1)
    #image_pub = rospy.Publisher("image_topic_2", Image, queue_size=1)
    image_sub = rospy.Subscriber("/kinect2/qhd/image_color", Image, callback)
    rospy.spin()

if __name__ == '__main__':
    main()
