import os
import numpy as np
import tensorflow as tf
import cv2 as cv
import time
from PIL import Image

IMG_DIR = '/media/storage/models/danet/test/'
#IMG_DIR = '/media/storage/datasets/I20191003/I20191003'

# Read the graph.
with tf.gfile.FastGFile('/media/storage/models/anno10/export/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Read and preprocess an image.
    start = time.time()
    count = 0
    for root, dirs, files in os.walk(IMG_DIR):
        for f in files:
            if '_gt' in f:
                continue
            ext = f.split('.')[-1]
            if ext.lower() not in ['jpg', 'png', 'jpeg', 'tif']:
                continue
            img = cv.imread(os.path.join(root, f))

            scale = max(img.shape[0] / 1024.0, img.shape[1] / 1024.0, 1.0)
            img = cv.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)))

            count += 1
            rows = img.shape[0]
            cols = img.shape[1]

            inp = img[:, :, [2, 1, 0]]  # BGR2RGB

            # Run the model
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                           feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

            # Visualize detected bounding boxes.
            num_detections = int(out[0][0])
            for i in range(num_detections):
                classId = int(out[3][0][i])
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                if score > 0.35:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                    cv.putText(img, str(score)[:4], (int(x), int(y)), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0))

            cv.imshow('TensorFlow', img)
            cv.waitKey()
    print('Took {} seconds for {} images'.format(time.time() - start, count))
