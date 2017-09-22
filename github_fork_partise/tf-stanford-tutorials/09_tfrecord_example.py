__author__ = 'fanfan'
from PIL import Image
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

IMAGE_PATH = "data/"

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def get_image_binary(filename):
    image = Image.open(filename)
    image = np.asarray(image,np.uint8)
    shape = np.array(image.shape,np.int32)
    return shape.tobytes(),image.tobytes()

def write_to_tfrecord(label,shape,binary_image,tfrecord_file):
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    example = tf.train.Example(features=tf.train.Features(feature={
        'label':_int64_feature(label),
        'shape':_bytes_feature(shape),
        'image':_bytes_feature(binary_image)
    }))

    writer.write(example.SerializeToString())
    writer.close()

def write_tfrecord(label,image_file,tfrecord_file):
    shape,binary_image = get_image_binary(image_file)
    write_to_tfrecord(label,shape,binary_image,tfrecord_file)


def read_from_tfrecord(filenames):
    tfrecord_file_queue = tf.train.string_input_producer(filenames,name='queue')
    reader = tf.TFRecordReader()
    _,tfrecord_serialized = reader.read(tfrecord_file_queue)

    tfrecord_features = tf.parse_single_example(tfrecord_serialized,features={
        'label':tf.FixedLenFeature([],tf.int64),
        'shape':tf.FixedLenFeature([],tf.string),
        'image':tf.FixedLenFeature([],tf.string),
    },name='features')


    image = tf.decode_raw(tfrecord_features['image'],tf.uint8)
    shape = tf.decode_raw(tfrecord_features['shape'],tf.int32)

    image = tf.reshape(image,shape)
    label = tfrecord_features['label']
    return label,shape,image

def read_tfrecord(tfrecod_file):
    label,shape,image = read_from_tfrecord([tfrecod_file])

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        label,image,shape = sess.run([label,image,shape])
        coord.request_stop()
        coord.join(threads)

    print(label)
    print(shape)
    plt.imshow(image)
    plt.show()


def main():
    label = 1
    image_file = IMAGE_PATH + 'friday.jpg'
    tfrecord_file = IMAGE_PATH + "friday.tfrecord"
    write_tfrecord(label,image_file,tfrecord_file)
    read_tfrecord(tfrecord_file)

if __name__ == '__main__':
    main()

