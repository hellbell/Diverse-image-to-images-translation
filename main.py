import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json

import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

flags = tf.app.flags
flags.DEFINE_integer("epoch", 200, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("crop_size", 128, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("image_size", 256, "The size of generated patches [64]")
flags.DEFINE_integer("output_size", 256, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

with tf.device('/gpu:1'):
    def main(_):
        pp.pprint(flags.FLAGS.__flags)

        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)
        if not os.path.exists(FLAGS.sample_dir):
            os.makedirs(FLAGS.sample_dir)

        aconfig = tf.ConfigProto()
        aconfig.gpu_options.allow_growth = True
        aconfig.allow_soft_placement = True

        with tf.Session(config=aconfig) as sess:
            dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, crop_size=FLAGS.crop_size, c_dim=FLAGS.c_dim,
                        dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir, sample_dir=FLAGS.sample_dir)

            if FLAGS.is_train:
                dcgan.train(FLAGS)
            else:
                dcgan.load(FLAGS.checkpoint_dir)
                #dcgan.test(FLAGS)

            if FLAGS.visualize:
                to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
                                              [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
                                              [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
                                              [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
                                              [dcgan.h4_w, dcgan.h4_b, None])

                # Below is codes for visualization
                OPTION = 2
                visualize(sess, dcgan, FLAGS, OPTION)

    if __name__ == '__main__':
        tf.app.run()
