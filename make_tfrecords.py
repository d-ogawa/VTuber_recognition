from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import glob
import random

TRAIN_IMAGE_DIR = './VTuber_train'
TEST_IMAGE_DIR = './VTuber_test'

OUTPUT_TRAIN_TFRECORD_DIR = './train_tfrecords'
OUTPUT_TEST_TFRECORD_DIR = './test_tfrecords'

# https://www.tdi.co.jp/miso/tensorflow-tfrecord-01

def make_tfrecords(file, label, base, outdir):

    print(base)
    tfrecords_filename = os.path.join(outdir, '{}.tfrecords'.format(base))
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    with Image.open(file) as image_object:  # (128x128x3) image

        image = np.array(image_object)
        height = image.shape[0]
        width = image.shape[1]
        dim = image.shape[2]

        example = tf.train.Example(features=tf.train.Features(feature={
                "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                "width" : tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                "dim"   : tf.train.Feature(int64_list=tf.train.Int64List(value=[dim])),
                "label" : tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                "image" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_object.tobytes()]))
                }))

    writer.write(example.SerializeToString())
    writer.close()


def divide_train_test(face, train_ratio):
    face_num = len(face)
    divide_idx = int(face_num * train_ratio)

    train, test = face[:divide_idx], face[divide_idx:]

    return train, test


random.seed(1)


KizunaAI_face = glob.glob('./face/KizunaAI/*.jpg')
random.shuffle(KizunaAI_face)
print('Num of KizunaAI faces : %d' %(len(KizunaAI_face)))
KizunaAI_train, KizunaAI_test = divide_train_test(KizunaAI_face, train_ratio=0.9)

MiraiAkari_face = glob.glob('./face/MiraiAkari/*.jpg')
random.shuffle(MiraiAkari_face)
print('Num of MiraiAkari faces : %d' %(len(MiraiAkari_face)))
MiraiAkari_train, MiraiAkari_test = divide_train_test(MiraiAkari_face, train_ratio=0.9)

KaguyaLuna_face = glob.glob('./face/KaguyaLuna/*.jpg')
random.shuffle(KaguyaLuna_face)
print('Num of KaguyaLuna faces : %d' %(len(KaguyaLuna_face)))
KaguyaLuna_train, KaguyaLuna_test = divide_train_test(KaguyaLuna_face, train_ratio=0.9)

Siro_face = glob.glob('./face/Siro/*.jpg')
random.shuffle(Siro_face)
print('Num of Siro faces : %d' %(len(Siro_face)))
Siro_train, Siro_test = divide_train_test(Siro_face, train_ratio=0.9)

NekoMas_face = glob.glob('./face/NekoMas/*.jpg')
random.shuffle(NekoMas_face)
print('Num of NekoMas faces : %d' %(len(NekoMas_face)))
NekoMas_train, NekoMas_test = divide_train_test(NekoMas_face, train_ratio=0.9)




# for train
if not os.path.exists(OUTPUT_TRAIN_TFRECORD_DIR):
    os.makedirs(OUTPUT_TRAIN_TFRECORD_DIR)

num = 0
for (label, files) in enumerate([KizunaAI_train, MiraiAkari_train, KaguyaLuna_train, Siro_train, NekoMas_train]):
    print(label, len(files))
    for file in files:
        base = '{:05}'.format(num)
        make_tfrecords(file, label, base, outdir=OUTPUT_TRAIN_TFRECORD_DIR)
        num += 1


# for test data
if not os.path.exists(OUTPUT_TEST_TFRECORD_DIR):
    os.makedirs(OUTPUT_TEST_TFRECORD_DIR)

num = 0
for (label, files) in enumerate([KizunaAI_test, MiraiAkari_test, KaguyaLuna_test, Siro_test, NekoMas_test]):
    print(label, len(files))
    for file in files:
        base = '{:05}'.format(num)
        make_tfrecords(file, label, base, outdir=OUTPUT_TEST_TFRECORD_DIR)
        num += 1
