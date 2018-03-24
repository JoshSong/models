#!/usr/bin/env python

import os
import json
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import Image

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

bridge = CvBridge()
image_pub = None

detection_graph = None
sess = None
category_index = None

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

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
        min_score_thresh=0.3,
        line_thickness=5)
    return image_np, scores, classes

def ros_service(req):
    print 'called'
    global video_capture, detection_graph, sess
    frame = video_capture.read()
    output_img, scores, classes = detect_objects(frame)
    success = scores[0][0] > 0.1
    #cv2.imshow('tensorflow', output_img)
    #cv2.waitKey(5000)
    #cv2.destroyAllWindows()

    return TriggerResponse(success=success, message=str(scores) + '\n' + str(classes))

def callback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    output_img, scores, classes = detect_objects(cv_image)

    cv2.imshow("Image window", output_img)
    cv2.waitKey(3)

    try:
        image_pub.publish(bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
        print(e)

def init_deep():
    global detection_graph, sess, category_index
    model_config = './models/detpri.json'
    model_details = json.load(open(model_config))

    # Path to frozen detection graph.
    # This is the actual model that is used for the object detection.
    MODEL_NAME = model_details['model_folder']
    MODEL_FILE = model_details['model_filename']
    PATH_TO_CKPT = os.path.join(FILE_DIR, 'models', MODEL_NAME, MODEL_FILE)

    # List of the strings that is used to add correct label for each box.
    LABELS_FILE = model_details['labels_filename']
    PATH_TO_LABELS = os.path.join(FILE_DIR, 'models', 'labels', LABELS_FILE)

    NUM_CLASSES = model_details['number_of_classes']

    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                max_num_classes=NUM_CLASSES,
                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load a (frozen) Tensorflow model into memory.
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

def main():
    global image_pub

    init_deep()

    rospy.init_node('tensorflow')
    s = rospy.Service('tensorflow_srv', Trigger, ros_service)

    image_pub = rospy.Publisher("image_topic_2", Image, queue_size=1)
    image_sub = rospy.Subscriber("/kinect2/hd/image_color", Image, callback)
    rospy.spin()

if __name__ == '__main__':
    main()