import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import time
import model
from tensorflow import gfile
from tensorflow import logging
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from operator import itemgetter


BATCH_SIZE = 100
NUM_EPOCHS = 1
LOGDIR = './logdir_byRNN/'

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNE = 3
TARGET_SIZE = 5
n_input = 128
n_steps = 128
n_hidden = 128

INPUT_TRAIN_TFRECORD = './train_tfrecords/*.tfrecords'
INPUT_TEST_TFRECORD = './test_tfrecords/*.tfrecords'


def RNN(x, BATCH_SIZE):
    x = tf.transpose(x, [1, 0, 2, 3])   # (IMAGE_HEIGHT, BATCH_SIZE, IMAGE_WIDTH, dim)
    x = tf.reshape(x, [-1, n_input])    # (128x100, 128)
    x = tf.split(x, n_steps, 0)         # (128x100) 128 tensors (sequences)

    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    weight = tf.Variable(tf.random_normal([n_hidden, TARGET_SIZE]))
    bias = tf.Variable(tf.random_normal([TARGET_SIZE]))

    return tf.matmul(outputs[-1], weight) + bias


def read_and_decode(filename_queue):

    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)

    features = tf.parse_single_example(
        value,
        features={'label'  : tf.FixedLenFeature([], tf.int64, default_value=0),
                  'image'  : tf.FixedLenFeature([], tf.string, default_value=""),
                  'height' : tf.FixedLenFeature([], tf.int64, default_value=0),
                  'width'  : tf.FixedLenFeature([], tf.int64, default_value=0),
                  'dim'    : tf.FixedLenFeature([], tf.int64, default_value=0)
        })

    label = tf.cast(features['label'], tf.int64)
    label = tf.one_hot(label, TARGET_SIZE)

    height = tf.cast(features['height'], tf.int64)
    width = tf.cast(features['width'], tf.int64)
    dim = tf.cast(features['dim'], tf.int64)

    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNE])
    image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, tf.float32)
    image = image / 255  # 画像データを、0～1の範囲に変換する

    return image, label


def inputs(batch_size, num_epochs, input_tfrecord):

    if not num_epochs:
        num_epochs = None

    with tf.name_scope('input'):

        files = gfile.Glob(input_tfrecord)
        files = sorted(files)

        print("files num : ", len(files))

        if not files:
            raise IOError("Unable to find training files. data_pattern='" +
                          input_tfrecord + "'.")
        logging.info("Number of training files: %s.", str(len(files)))

        filename_queue = tf.train.string_input_producer(files,
                                                        num_epochs=num_epochs,
                                                        shuffle=False)

        image, label = read_and_decode(filename_queue)

        print("image     :", image.shape)
        print("label      :", label.shape)

        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=10,
            capacity=10000 + 15 * batch_size,
            min_after_dequeue=10000,
            allow_smaller_final_batch=False # True --> error ...
            )

        return image_batch, label_batch


def save_result(image_batch_step, softmax_step, step):
    label = ['KizunaAI', 'MiraiAkari', 'KaguyaLuna', 'Siro', 'NekoMas']

    for i, (image, softmax) in enumerate(zip(image_batch_step, softmax_step)):
        label_tuples = []
        for (l, s) in zip(label, softmax):
            label_tuples.append((l, s))
        label_tuples = sorted(label_tuples, key=itemgetter(1), reverse=True)

        image = image * 255
        image = image.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
        # image = np.abs(np.fft.rfft2(image, axes=(0, 1)))
        image = Image.fromarray(np.uint8(image))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/home/ogawa/.fonts/Ubuntu-L.ttf", 10)

        for (j, r) in enumerate(label_tuples):
            l, s = r
            draw.text((10, j * 10), l + ' : {:.3f}'.format(s), fill=(0), font=font)

        image.save('./RESULT_byRNN/' + str(step) + '-' + str(i) + '.jpg')


def load_pretrain(sess):
    pre_train_saver.restore(sess, './logdir/model.ckpt-0.meta')


if __name__ == "__main__":

    with tf.Graph().as_default():

        print('Reading batches...')
        image_batch, label_batch = inputs(batch_size=BATCH_SIZE,
                                          num_epochs=NUM_EPOCHS,
                                          input_tfrecord=INPUT_TEST_TFRECORD)

        print('build models...')
        y_conv = RNN(image_batch, BATCH_SIZE)
        softmax = tf.nn.softmax(y_conv)
        # with tf.name_scope('test'):
        #     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=label_batch))

        global_step = tf.Variable(0, trainable=False)

        # calculate accuracy
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(label_batch, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sv = tf.train.Supervisor(logdir=LOGDIR,
                                 global_step=global_step)
                                 # init_fn=load_pretrain)
                                 # global_step=global_step,
                                 # save_summaries_secs=10,
                                 # save_model_secs=120)
        # init = tf.initialize_all_variables()
        # sess.run(init) #変数を初期化して実行

        with sv.managed_session(config=config) as sess:
            print('start loop...' + datetime.now().strftime("%Y%m%d-%H%M%S"))

            try:
                step = 0
                accu_all = []
                while not sv.should_stop():
                    start_time = time.time()

                    accu_step, softmax_step, image_batch_step, g_step \
                            = sess.run([accuracy, softmax, image_batch, global_step])
                    accu_all.append(accu_step)
                    print(softmax_step[:10])
                    print(image_batch_step.shape)
                    save_result(image_batch_step, softmax_step, step)

                    duration = time.time() - start_time

                    print('Step test %04d: accu = %07.4f (%02.3f sec)' %(step,
                                                                         accu_step,
                                                                         duration))
                    step += 1

            except tf.errors.OutOfRangeError:
                print('Done testing for %d epochs, %d steps.' %
                      (NUM_EPOCHS, step))

            accu_all_mean = np.array(accu_all).mean()
            print("accu_all_mean : ", accu_all_mean)

            sv.Stop()

        print('End loop...' + datetime.now().strftime("%Y%m%d-%H%M%S"))
