from __future__ import division
import os
import time
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
import numpy as np

from six.moves import xrange

from itertools import chain
import tensorflow.contrib.slim as slim
import scipy.io

from collections import namedtuple

from ops import *
from utils import *
from module import *

class DCGAN(object):
    def __init__(self, sess, is_crop=True,
                 batch_size=64, sample_size = 64, crop_size=128, image_size=64,
                 y_dim=None, z_dim=128, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None, embed_dim=128):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """

        # current setting
        #

        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size

        self.crop_size = crop_size

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.embed_dim = embed_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.c_dim = c_dim

        OPTIONS = namedtuple('OPTIONS', 'batch_size patch_size df_dim gf_dim embed_dim c_dim')
        self.options = OPTIONS._make((self.batch_size, self.image_size, self.df_dim, self.gf_dim, self.embed_dim, self.c_dim))

        # self.generator_from_z = generator_from_z
        # self.generator_pix2pix = generator_pix2pix
        # self.generator = generator
        # self.generator_partial = generator_partial
        # self.generator_rest = generator_rest
        # self.discriminator = discriminator

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.imA = tf.placeholder(tf.float32, [self.batch_size] + [self.image_size, self.image_size, self.c_dim])
        self.imB = tf.placeholder(tf.float32, [self.batch_size] + [self.image_size, self.image_size, self.c_dim])
        self.is_training = tf.placeholder(tf.bool)

        self.z = tf.placeholder(tf.float32, [None, self.z_dim])
        self.z_sum = tf.summary.histogram("z", self.z)

        g_recon_d3 = generator_partial(self.imA, 0., self.options, self.is_training, False, "g_branch")
        g_adv_d3 = generator_partial(self.imA, self.z, self.options, self.is_training, True, "g_branch")
        self.g_recon = generator_rest(g_recon_d3, self.options, self.is_training, False, "recon")
        self.g_adv = generator_rest(g_adv_d3, self.options, self.is_training, False, "adv")
        self.g_recon_adv = generator_pix2pix(self.imA, self.z, self.options, self.is_training, False, "g_nobranch")

        d1_real, d1_real_logits = discriminator(self.imB, self.options, self.is_training, False, "d_branch")
        d1_gen, d1_gen_logits = discriminator(self.g_adv, self.options, self.is_training, True, "d_branch")

        d2_real, d2_real_logits = discriminator(self.imB, self.options, self.is_training, False, "d_nobranch")
        d2_gen, d2_gen_logits = discriminator(self.g_recon_adv, self.options, self.is_training, True, "d_nobranch")

        d1_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d1_real_logits, labels=tf.ones_like(d1_real))
            + tf.nn.sigmoid_cross_entropy_with_logits(logits=d1_gen_logits, labels=tf.zeros_like(d1_gen)))

        d2_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d2_real_logits, labels=tf.ones_like(d2_real))
            + tf.nn.sigmoid_cross_entropy_with_logits(logits=d2_gen_logits, labels=tf.zeros_like(d2_gen)))

        self.d1_loss = d1_loss
        self.d2_loss = d2_loss
        self.d_loss = d1_loss + d2_loss

        g1_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d1_gen_logits, labels=tf.ones_like(d1_gen)))
        g2_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d2_gen_logits, labels=tf.ones_like(d2_gen)))

        g1_l1_loss = tf.reduce_mean(tf.abs(self.g_recon - self.imB))
        g2_l1_loss = tf.reduce_mean(tf.abs(self.g_recon_adv - self.imB))

        feat_err = 0
        self.g1_loss = g1_adv_loss + 1. * g1_l1_loss + feat_err
        self.g2_loss = g2_adv_loss + 1. * g2_l1_loss
        self.g_loss = self.g1_loss + self.g2_loss

        self.d1_loss_sum = tf.summary.scalar("d1_loss", d1_loss)
        self.d2_loss_sum = tf.summary.scalar("d2_loss", d2_loss)
        self.g1_loss_sum = tf.summary.scalar("g1_loss", g1_adv_loss)
        self.g2_loss_sum = tf.summary.scalar("g2_loss", g2_adv_loss)
        self.recon1_loss_sum = tf.summary.scalar("recon1_loss", g1_l1_loss)
        self.recon2_loss_sum = tf.summary.scalar("recon2_loss", g2_l1_loss)
        self.feat_loss_sum = tf.summary.scalar("feat_loss", feat_err)

        t_vars = tf.trainable_variables()

        self.g1_vars = [var for var in t_vars if 'g_branch/' in var.name] +\
                       [var for var in t_vars if 'recon/' in var.name] +\
                       [var for var in t_vars if 'adv/' in var.name]
        self.g2_vars = [var for var in t_vars if 'g_nobranch/' in var.name]
        self.d1_vars = [var for var in t_vars if 'd_branch/' in var.name]
        self.d2_vars = [var for var in t_vars if 'd_nobranch/' in var.name]

        self.saver = tf.train.Saver()

    def resize_image(self, im, newSize):
        return scipy.misc.imresize(im, [newSize, newSize])

    def train(self, config):
        """Train DCGAN"""
        # data = glob(os.path.join("/data1/facades/img/*.jpg"))
        data = glob(os.path.join("/data1/pix2pix/edges2shoes/train/*.jpg"))
        # data = glob(os.path.join("/data1/celebA/img/img_align_celeba_png/*.png"))
        data = np.array(sorted(data))

        g1_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g1_loss, var_list=self.g1_vars)
        g2_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g2_loss, var_list=self.g2_vars)
        d1_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d1_loss, var_list=self.d1_vars)
        d2_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d2_loss, var_list=self.d2_vars)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.g_sum = tf.summary.merge([self.z_sum, self.g1_loss_sum, self.g2_loss_sum, self.feat_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.d1_loss_sum, self.d2_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        np.random.seed(220)
        sample_z1 = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
        np.random.seed(221)
        sample_z2 = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
        np.random.seed(222)
        sample_z3 = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))

        # prepare sample images
        sample_files = data[0:self.batch_size]
        sample_images = np.array([load_data(sample_file) for sample_file in sample_files]).astype(np.float32)
        sample_imA = sample_images[:, :, :, :self.c_dim]
        sample_imB = sample_images[:, :, :, self.c_dim:]
        save_images(sample_imA[0:self.sample_size], [8, 8], './{}/sampleA.png'.format(config.sample_dir))
        save_images(sample_imB[0:self.sample_size], [8, 8], './{}/sampleB.png'.format(config.sample_dir))

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # all
        for epoch in xrange(config.epoch):
        # for epoch in xrange(50, config.epoch):
            # np.random.seed(epoch)
            np.random.shuffle(data)
            batch_idxs = min(len(data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[xrange(idx * config.batch_size, (idx + 1) * config.batch_size)]
                batch_images = np.array([load_data(batch_file) for batch_file in batch_files]).astype(np.float32)
                batch_imA = batch_images[:, :, :, :self.c_dim]
                batch_imB = batch_images[:, :, :, self.c_dim:]
                batch_z1 = np.random.normal(0, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # Update G network
                _, summary_str = self.sess.run([d1_optim, self.d_sum],
                                   feed_dict={self.imA: batch_imA, self.imB: batch_imB, self.z: batch_z1, self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                _, summary_str = self.sess.run([d2_optim, self.d_sum],
                                   feed_dict={self.imA: batch_imA, self.imB: batch_imB, self.z: batch_z1, self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                _, summary_str = self.sess.run([g1_optim, self.g_sum],
                                   feed_dict={self.imA: batch_imA, self.imB: batch_imB, self.z: batch_z1, self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                _, summary_str = self.sess.run([g1_optim, self.g_sum],
                                   feed_dict={self.imA: batch_imA, self.imB: batch_imB, self.z: batch_z1, self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                _, summary_str = self.sess.run([g2_optim, self.g_sum],
                                   feed_dict={self.imA: batch_imA, self.imB: batch_imB, self.z: batch_z1, self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                _, summary_str = self.sess.run([g2_optim, self.g_sum],
                                   feed_dict={self.imA: batch_imA, self.imB: batch_imB, self.z: batch_z1, self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f"
                      % (epoch, idx, batch_idxs, time.time() - start_time))

                if np.mod(counter, 100) == 1:
                    imA, g_recon, g_adv1, g_recon_adv1 = self.sess.run(
                        [self.imA, self.g_recon, self.g_adv, self.g_recon_adv],
                        feed_dict={self.imA: sample_imA, self.imB: sample_imB, self.z: sample_z1, self.is_training: False})
                    g_adv2, g_recon_adv2 = self.sess.run(
                        [self.g_adv, self.g_recon_adv],
                        feed_dict={self.imA: sample_imA, self.imB: sample_imB, self.z: sample_z2, self.is_training: False})
                    g_adv3, g_recon_adv3 = self.sess.run(
                        [self.g_adv, self.g_recon_adv],
                        feed_dict={self.imA: sample_imA, self.imB: sample_imB, self.z: sample_z3, self.is_training: False})
                    nS = 12
                    concatIm = np.concatenate((imA[:nS], g_recon[:nS],
                                               g_recon_adv1[:nS], g_recon_adv2[:nS], g_recon_adv3[:nS],
                                               g_adv1[:nS], g_adv2[:nS], g_adv3[:nS],
                                               sample_imB[:nS]))
                    save_images(concatIm, [9, nS], './{}/train_pair_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False