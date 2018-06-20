import tensorflow as tf
import PIL.Image
import os
import io

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('output_dir', '/home/Staff/uqjsong2/data', 'Path to output TFRecord')
flags.DEFINE_string('input_dir', '/home/Staff/uqjsong2/data/flickr_32_v2', 'Path to FlickrLogos-32')
FLAGS = flags.FLAGS

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
label_map_path = os.path.join(dir_path, 'data', 'flickr32_label_map.pbtxt')
input_pb = label_map_util.load_labelmap(label_map_path)
label_map = {}
for i in input_pb.item:
    label_map[i.display_name] = i.id

def create_tf_example(logo, img_path, bbox_path):
    img_path = str(img_path)
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    height = image.size[1]
    width = image.size[0]

    xmins = []    # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []    # List of normalized right x coordinates in bounding box (1 per box)
    ymins = []    # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []    # List of normalized bottom y coordinates in bounding box (1 per box)

    fwidth = float(width)
    fheight = float(height)
    texts = []
    labels = []
    if os.path.exists(bbox_path):
        with open(bbox_path) as fp:
            first = True
            for line in fp:
                if first:
                    first = False
                    continue
                s = line.split()
                x = float(s[0])
                y = float(s[1])
                bbwidth = float(s[2])
                bbheight = float(s[3])

                xmins.append(x / fwidth)
                xmaxs.append((x + bbwidth) / fwidth)
                ymins.append(y / fheight)
                ymaxs.append((y + bbheight) / fheight)

        texts = [logo] * len(xmins)
        labels = [label_map[logo]] * len(xmins)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(img_path),
            'image/source_id': dataset_util.bytes_feature(img_path),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(texts),
            'image/object/class/label': dataset_util.int64_list_feature(labels),
    }))
    return tf_example


def main(_):
    jpg_dir = os.path.join(FLAGS.input_dir, 'classes', 'jpg')
    masks_dir = os.path.join(FLAGS.input_dir, 'classes', 'masks')

    train = []
    with open(os.path.join(FLAGS.input_dir, 'trainvalset.txt')) as fp:
        for line in fp:
            line = line.strip()
            train.append(line.split(','))

    writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_dir, 'flickrlogos_32_train.record'))
    for logo, img in train:
        img_path = os.path.join(jpg_dir, logo, img)
        bbox_path = os.path.join(masks_dir, logo.lower(), img + '.bboxes.txt')
        tf_example = create_tf_example(logo, img_path, bbox_path)
        writer.write(tf_example.SerializeToString())
    writer.close()

    test = []
    with open(os.path.join(FLAGS.input_dir, 'testset.txt')) as fp:
        for line in fp:
            line = line.strip()
            test.append(line.split(','))

    writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_dir, 'flickrlogos_32_test.record'))
    for logo, img in test:
        img_path = os.path.join(jpg_dir, logo, img)
        bbox_path = os.path.join(masks_dir, logo.lower(), img + '.bboxes.txt')
        tf_example = create_tf_example(logo, img_path, bbox_path)
        writer.write(tf_example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    tf.app.run()
