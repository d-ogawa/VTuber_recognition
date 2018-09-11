import numpy as np
import tensorflow as tf
import time
import model
from tensorflow import gfile
from tensorflow import logging
from datetime import datetime


BATCH_SIZE = 100
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
LOGDIR = './logdir_byMT/'

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNE = 3
TARGET_SIZE = 5

INPUT_TRAIN_TFRECORD = './train_tfrecords/*.tfrecords'
INPUT_TEST_TFRECORD = './test_tfrecords/*.tfrecords'


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

    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, TARGET_SIZE)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    dim = tf.cast(features['dim'], tf.int32)

    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = image / 255  # 画像データを、0～1の範囲に変換する
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNE])

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
                                                        shuffle=True)

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

        tf.summary.image('input', image_batch)

        return image_batch, label_batch, len(files)


if __name__ == "__main__":

    with tf.Graph().as_default():

        print('Reading batches...')
        image_batch, label_batch, file_num = inputs(batch_size=BATCH_SIZE,
                                                    num_epochs=NUM_EPOCHS,
                                                    input_tfrecord=INPUT_TRAIN_TFRECORD)

        print('build models...')
        y_conv = model.inference(image_batch, BATCH_SIZE, is_training=True)

        with tf.name_scope('train'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=label_batch))
            tf.summary.scalar('loss', loss)

        global_step = global_step = tf.train.get_or_create_global_step()
        k = 100 * 10**3 # 100k steps
        learning_rate = tf.train.inverse_time_decay(LEARNING_RATE, global_step, k, 1, staircase=True)

        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

        # calculate accuracy
        with tf.name_scope('test'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(label_batch, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        train_step_summary = tf.summary.merge_all()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir     =LOGDIR,
                                                  save_secs          =120,
                                                  save_steps         =None,
                                                  saver              =tf.train.Saver(),
                                                  checkpoint_basename='model.ckpt',
                                                  scaffold           =None,
                                                  listeners          =None
                                                  )

        summary_hook = tf.train.SummarySaverHook(save_steps    =None,
                                                 save_secs     =10,
                                                 output_dir    =LOGDIR,
                                                 summary_writer=None,
                                                 scaffold      =None,
                                                 summary_op    =train_step_summary)

        steps_hook = tf.train.StopAtStepHook(last_step=file_num / BATCH_SIZE * NUM_EPOCHS) # 1000epochs -> steps

        with tf.train.MonitoredTrainingSession(checkpoint_dir=LOGDIR,
                                               hooks=[saver_hook, summary_hook, steps_hook],
                                               config=config) as sess:

            print('start loop...' + datetime.now().strftime("%Y%m%d-%H%M%S"))

            try:
                step = 0
                while not sess.should_stop():
                    start_time = time.time()

                    _, loss_value, g_step = sess.run([train_step, loss, global_step])

                    duration = time.time() - start_time

                    print('Step train %04d     : loss = %07.4f (%02.3f sec)' % (g_step,
                                                                              loss_value,
                                                                              duration))

                    if step % 100 == 0:
                        est_accuracy, est_y, gt_y = sess.run([accuracy, y_conv, label_batch])
                        print("Accuracy (for test data): {:5.2f}".format(est_accuracy))
                        print("True Label:", np.argmax(gt_y[0:15,], 1))
                        print("Est Label :", np.argmax(est_y[0:15, ], 1))

                    step += 1

            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' %
                      (NUM_EPOCHS, step))


        print('End loop...' + datetime.now().strftime("%Y%m%d-%H%M%S"))
