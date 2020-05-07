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
flags.DEFINE_string('imgs_dir', 'syn_patent6', 'Path to imgs directory')
flags.DEFINE_string('info_path', 'syn_patent6_info.json', 'Path to info json')
flags.DEFINE_string('output_path', 'output.record', 'Path to output TFRecord')
flags.DEFINE_string('img_ext', '.png', 'Image extension')
flags.DEFINE_boolean('rotated_90', False, 'Is image rotated 90 degrees ccw')
FLAGS = flags.FLAGS


def dict_to_tf_example(data, img_path, label_map_dict):
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
    for obj in data:
        if obj['text'] == '' or obj['text'] is None:
            continue

        tl = obj['box top left']
        bs = obj['box size']
        br = (tl[0] + bs[0], tl[1] + bs[1])

        xmin.append(float(tl[0])/width)
        ymin.append(float(tl[1])/height)
        xmax.append(float(br[0])/width)
        ymax.append(float(br[1])/height)

        if FLAGS.rotated_90:
            classes_text.append('part90'.encode('utf8'))
            classes.append(label_map_dict['part90'])
        else:
            classes_text.append('part'.encode('utf8'))
            classes.append(label_map_dict['part'])

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

    infos = json.load(open(FLAGS.info_path))

    if FLAGS.rotated_90:
        print('rotated 90')

    count = 0
    for id, info in infos.iteritems():
        if count % 100 == 0:
            logging.info('{}/{}'.format(count, len(infos)))
        count += 1

        img_path = os.path.join(FLAGS.imgs_dir, id + FLAGS.img_ext)

        tf_example = dict_to_tf_example(info, img_path, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
