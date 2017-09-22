__author__ = 'fanfan'
import tensorflow as tf

DATA_PATH = 'data/heart.csv'
BATCH_SIZE = 2
N_FEATURES = 9


def batch_generator(filenames):
    """ filenames is the list of files you want to read from.
    In this case, it contains only heart.csv
    """
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TextLineReader(skip_header_lines=1)
    _,value = reader.read(filename_queue)

    record_defaults = [[1.0] for _ in range(N_FEATURES)]
    record_defaults[4] = ['']
    record_defaults.append([1])

    content = tf.decode_csv(value,record_defaults=record_defaults)
    content[4] = tf.cond(tf.equal(content[4],tf.constant('Present')),lambda : tf.constant(1.0),lambda :tf.constant(0.0))

    features = tf.stack(content[:N_FEATURES])
    label = content[-1]

    min_after_dequeue = 10 * BATCH_SIZE
    capacity = 20 * BATCH_SIZE

    data_batch,laebl_batch = tf.train.shuffle_batch([features,label],batch_size=BATCH_SIZE,capacity=capacity,min_after_dequeue=min_after_dequeue)
    return data_batch,laebl_batch

def generate_batches(data_batch,label_batch):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        for _ in range(10):
            features,labels = sess.run([data_batch,label_batch])
            print(features)

        coord.request_stop()
        coord.join(threads)

def main():
    data_batch,label_batch = batch_generator([DATA_PATH])
    generate_batches(data_batch,label_batch)

if __name__ == '__main__':
    main()
