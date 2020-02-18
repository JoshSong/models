"""Convert synthetic label bounding boxes to TFRecord for object_detection."""
import hashlib
import io
import logging
import os
import json

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

this_dir = os.path.dirname(os.path.abspath(__file__))
label_map_path = os.path.join(os.path.dirname(this_dir), 'data', 'anno_label_map.pbtxt')


flags = tf.app.flags
flags.DEFINE_string('imgs_dir', None, 'Path to imgs directory')
flags.DEFINE_string('info_dir', None, 'Path to info directory')
flags.DEFINE_string('output_path', None, 'Path to output TFRecord')
flags.DEFINE_string('img_ext', '.jpg', 'Image extension')
FLAGS = flags.FLAGS


def dict_to_tf_example(lines, img_path, label_map_dict):
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded = fid.read()
    encoded_io = io.BytesIO(encoded)
    image = PIL.Image.open(encoded_io)
    key = hashlib.sha256(encoded).hexdigest()
    filename = os.path.basename(img_path)
    width, height = image.size

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    for line in lines:
        s = line.split()
        npoints = int(s[0])
        xpoints = []
        ypoints = []
        for i in range(npoints):
            xpoints.append(float(s[i * 2 + 1]))
            ypoints.append(float(s[i * 2 + 2]))

        xmin.append(min(xpoints)/width)
        ymin.append(min(ypoints)/height)
        xmax.append(max(xpoints)/width)
        ymax.append(max(ypoints)/height)
        if max(xmin) > 1.0 or max(ymin) > 1.0 or max(xmax) > 1.0 or max(ymax) > 1.0:
            import pdb;pdb.set_trace()
        classes_text.append('label'.encode('utf8'))
        classes.append(label_map_dict['label'])

    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded),
            'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example


def main(_):


    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(label_map_path)

    count = 0
    for f in os.listdir(FLAGS.info_dir):
        lines = []
        with open(os.path.join(FLAGS.info_dir, f)) as fp:
            for line in fp:
                lines.append(line.strip())
        lines = lines[1:]

        img_path = os.path.join(FLAGS.imgs_dir, f.rsplit('.', 1)[0] + FLAGS.img_ext)

        tf_example = dict_to_tf_example(lines, img_path, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
