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

from ops import *
from utils import *

def generator_pix2pix(image, z, options, is_training=True, reuse=False, var_sc="gen_unet"):
    s = options.patch_size
    s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(s / 64), int(s / 128)

    conv_init_params = tf.truncated_normal_initializer(stddev=0.02)
    fully_init_params = tf.random_normal_initializer(stddev=0.02)
    bias_init_params = tf.constant_initializer(0.0)
    bn_init_params = {'beta': tf.constant_initializer(0.), 'gamma': tf.random_normal_initializer(1., 0.02)}
    bn_params = {'is_training': is_training, 'decay': 0.9, 'epsilon': 1e-5, 'param_initializers': bn_init_params,
                 'updates_collections': None}
    with tf.variable_scope(var_sc) as scope:
        if reuse:
            scope.reuse_variables()

        e0 = slim.conv2d(image, options.df_dim, [5, 5], stride=2, activation_fn=lrelu,
                         weights_initializer=conv_init_params, biases_initializer=bias_init_params, scope='e0')
        # e0 is (128 x 128)
        e1 = slim.conv2d(e0, options.df_dim * 2, [5, 5], stride=2, activation_fn=lrelu,
                         weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                         normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='e1')
        # e1 is (64 x 64)
        e2 = slim.conv2d(e1, options.df_dim * 4, [5, 5], stride=2, activation_fn=lrelu,
                         weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                         normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='e2')
        # e2 is (32 x 32)
        e3 = slim.conv2d(e2, options.df_dim * 8, [5, 5], stride=2, activation_fn=lrelu,
                         weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                         normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='e3')
        # e3 is (16 x 16)
        e4 = slim.conv2d(e3, options.df_dim * 8, [5, 5], stride=2, activation_fn=lrelu,
                         weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                         normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='e4')
        # e4 is (8 x 8)
        e5 = slim.conv2d(e4, options.df_dim * 8, [5, 5], stride=2, activation_fn=lrelu,
                         weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                         normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='e5')
        # e5 is (4 x 4)
        e6 = slim.conv2d(e5, options.df_dim * 8, [5, 5], stride=2, activation_fn=lrelu,
                         weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                         normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='e6')
        # e6 is (2 x 2)
        e7 = slim.fully_connected(tf.reshape(e6, [options.batch_size, -1]), options.embed_dim, activation_fn=None,
                                  weights_initializer=fully_init_params, biases_initializer=bias_init_params,
                                  scope='e7')
        # e7 is (1 x 1)

        # e4 = tf.concat([e4, z], axis=1)
        e7 = tf.nn.l2_normalize(e7, 1) + z
        d0 = slim.fully_connected(e7, options.gf_dim * 8 * s128 * s128, activation_fn=None,
                                  weights_initializer=fully_init_params, biases_initializer=bias_init_params,
                                  scope='lin0')
        d0 = tf.reshape(d0, [options.batch_size, s128, s128, options.gf_dim * 8])
        d0 = slim.batch_norm(d0, decay=0.9, param_initializers=bn_init_params, activation_fn=tf.nn.relu, scope='lin1')
        d0 = tf.nn.dropout(d0, 0.5)
        d0 = tf.concat([d0, e6], axis=3)
        # d0 is (2 x 2)
        d1 = slim.conv2d_transpose(d0, options.gf_dim * 8, [5, 5], stride=2, activation_fn=tf.nn.relu,
                                   weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                   normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='dconv1')
        d1 = tf.nn.dropout(d1, 0.5)
        d1 = tf.concat([d1, e5], axis=3)
        # d1 is (4 x 4)
        d2 = slim.conv2d_transpose(d1, options.gf_dim * 8, [5, 5], stride=2, activation_fn=tf.nn.relu,
                                   weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                   normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='dconv2')
        d2 = tf.nn.dropout(d2, 0.5)
        d2 = tf.concat([d2, e4], axis=3)
        # d2 is (8 x 8)
        d3 = slim.conv2d_transpose(d2, options.gf_dim * 8, [5, 5], stride=2, activation_fn=tf.nn.relu,
                                   weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                   normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='dconv3')
        d3 = tf.concat([d3, e3], axis=3)
        # d3 is (16 x 16)
        d4 = slim.conv2d_transpose(d3, options.gf_dim * 4, [5, 5], stride=2, activation_fn=tf.nn.relu,
                                   weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                   normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='dconv4')
        d4 = tf.concat([d4, e2], axis=3)
        # d4 is (32 x 32)
        d5 = slim.conv2d_transpose(d4, options.gf_dim * 2, [5, 5], stride=2, activation_fn=tf.nn.relu,
                                   weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                   normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='dconv5')
        d5 = tf.concat([d5, e1], axis=3)
        # d5 is (64 x 64)
        d6 = slim.conv2d_transpose(d5, options.gf_dim * 1, [5, 5], stride=2, activation_fn=tf.nn.relu,
                                   weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                   normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='dconv6')
        d6 = tf.concat([d6, e0], axis=3)
        # d6 is (128 x 128)
        gen = slim.conv2d_transpose(d6, options.c_dim, [5, 5], stride=2, activation_fn=tf.nn.tanh,
                                    weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                    scope='dconv7')
        # gen is (256 x 256)

    return gen

def generator_partial(image, z, options, is_training=True, reuse=False, var_sc="gen_partial"):
    s = options.patch_size
    s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(s / 64), int(s / 128)

    conv_init_params = tf.truncated_normal_initializer(stddev=0.02)
    fully_init_params = tf.random_normal_initializer(stddev=0.02)
    bias_init_params = tf.constant_initializer(0.0)
    bn_init_params = {'beta': tf.constant_initializer(0.), 'gamma': tf.random_normal_initializer(1., 0.02)}
    bn_params = {'is_training': is_training, 'decay': 0.9, 'epsilon': 1e-5, 'param_initializers': bn_init_params,
                 'updates_collections': None}
    with tf.variable_scope(var_sc) as scope:
        if reuse:
            scope.reuse_variables()

        e0 = slim.conv2d(image, options.df_dim, [5, 5], stride=2, activation_fn=lrelu,
                         weights_initializer=conv_init_params, biases_initializer=bias_init_params, scope='e0')
        # e0 is (128 x 128)
        e1 = slim.conv2d(e0, options.df_dim * 2, [5, 5], stride=2, activation_fn=lrelu,
                         weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                         normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='e1')
        # e1 is (64 x 64)
        e2 = slim.conv2d(e1, options.df_dim * 4, [5, 5], stride=2, activation_fn=lrelu,
                         weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                         normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='e2')
        # e2 is (32 x 32)
        e3 = slim.conv2d(e2, options.df_dim * 8, [5, 5], stride=2, activation_fn=lrelu,
                         weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                         normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='e3')
        # e3 is (16 x 16)
        e4 = slim.conv2d(e3, options.df_dim * 8, [5, 5], stride=2, activation_fn=lrelu,
                         weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                         normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='e4')
        # e4 is (8 x 8)
        e5 = slim.conv2d(e4, options.df_dim * 8, [5, 5], stride=2, activation_fn=lrelu,
                         weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                         normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='e5')
        # e5 is (4 x 4)
        e6 = slim.conv2d(e5, options.df_dim * 8, [5, 5], stride=2, activation_fn=lrelu,
                         weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                         normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='e6')
        # e6 is (2 x 2)
        e7 = slim.fully_connected(tf.reshape(e6, [options.batch_size, -1]), options.embed_dim, activation_fn=None,
                                  weights_initializer=fully_init_params, biases_initializer=bias_init_params,
                                  scope='e7')
        # e7 is (1 x 1)

        # e4 = tf.concat([e4, z], axis=1)
        e7 = tf.nn.l2_normalize(e7, 1) + z
        d0 = slim.fully_connected(e7, options.gf_dim * 8 * s128 * s128, activation_fn=None,
                                  weights_initializer=fully_init_params, biases_initializer=bias_init_params,
                                  scope='lin0')
        d0 = tf.reshape(d0, [options.batch_size, s128, s128, options.gf_dim * 8])
        d0 = slim.batch_norm(d0, decay=0.9, param_initializers=bn_init_params, activation_fn=tf.nn.relu, scope='lin1')
        d0 = tf.nn.dropout(d0, 0.5)
        d0 = tf.concat([d0, e6], axis=3)
        # d0 is (2 x 2)
        d1 = slim.conv2d_transpose(d0, options.gf_dim * 8, [5, 5], stride=2, activation_fn=tf.nn.relu,
                                   weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                   normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='dconv1')
        d1 = tf.nn.dropout(d1, 0.5)
        d1 = tf.concat([d1, e5], axis=3)
        # d1 is (4 x 4)
        d2 = slim.conv2d_transpose(d1, options.gf_dim * 8, [5, 5], stride=2, activation_fn=tf.nn.relu,
                                   weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                   normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='dconv2')
        d2 = tf.nn.dropout(d2, 0.5)
        d2 = tf.concat([d2, e4], axis=3)
        # d2 is (8 x 8)
        d3 = slim.conv2d_transpose(d2, options.gf_dim * 8, [5, 5], stride=2, activation_fn=tf.nn.relu,
                                   weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                   normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='dconv3')
        d3 = tf.concat([d3, e3], axis=3)
        # d3 is (16 x 16)

    return d3

def generator_rest(d3, options, is_training=True, reuse=False, var_sc="gen_rest"):
    s = options.patch_size
    s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(s / 64), int(s / 128)

    conv_init_params = tf.truncated_normal_initializer(stddev=0.02)
    fully_init_params = tf.random_normal_initializer(stddev=0.02)
    bias_init_params = tf.constant_initializer(0.0)
    bn_init_params = {'beta': tf.constant_initializer(0.), 'gamma': tf.random_normal_initializer(1., 0.02)}
    bn_params = {'is_training': is_training, 'decay': 0.9, 'epsilon': 1e-5, 'param_initializers': bn_init_params,
                 'updates_collections': None}
    with tf.variable_scope(var_sc) as scope:
        if reuse:
            scope.reuse_variables()

        d4 = slim.conv2d_transpose(d3, options.gf_dim * 4, [5, 5], stride=2, activation_fn=tf.nn.relu,
                                   weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                   normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='dconv4')
        # d4 = tf.concat([d4, e3], axis=3)
        # d4 is (32 x 32)
        d5 = slim.conv2d_transpose(d4, options.gf_dim * 2, [5, 5], stride=2, activation_fn=tf.nn.relu,
                                   weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                   normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='dconv5')
        # d5 = tf.concat([d5, e3], axis=3)
        # d3 is (64 x 64)
        d6 = slim.conv2d_transpose(d5, options.gf_dim * 1, [5, 5], stride=2, activation_fn=tf.nn.relu,
                                   weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                   normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='dconv6')
        # d6 = tf.concat([d6, e2], axis=3)
        # d3 is (128 x 128)
        gen = slim.conv2d_transpose(d6, options.c_dim, [5, 5], stride=2, activation_fn=tf.nn.tanh,
                                    weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                    scope='dconv7')
        # gen is (256 x 256)

    return gen

def discriminator(image, options, is_training=True, reuse=False, var_sc="d"):
    conv_init_params = tf.truncated_normal_initializer(stddev=0.02)
    fully_init_params = tf.random_normal_initializer(stddev=0.02)
    bias_init_params = tf.constant_initializer(0.0)
    bn_init_params = {'beta': tf.constant_initializer(0.), 'gamma': tf.random_normal_initializer(1., 0.02)}
    bn_params = {'is_training': is_training, 'decay': 0.9, 'epsilon': 1e-5, 'param_initializers': bn_init_params,
                 'updates_collections': None}
    with tf.variable_scope(var_sc) as scope:
        if reuse:
            scope.reuse_variables()

        # image is (256 x 256)
        h0 = slim.conv2d(image, options.df_dim, [5, 5], stride=2, activation_fn=lrelu,
                              weights_initializer=conv_init_params, biases_initializer=bias_init_params, scope='conv0')
        # e0 is (128 x 128)
        h1 = slim.conv2d(h0, options.df_dim * 2, [5, 5], stride=2, activation_fn=lrelu,
                              weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                              normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='conv1')
        # e1 is (64 x 64)
        h2 = slim.conv2d(h1, options.df_dim * 4, [5, 5], stride=2, activation_fn=lrelu,
                              weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                              normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='conv2')
        # e2 is (32 x 32)
        h3 = slim.conv2d(h2, options.df_dim * 8, [5, 5], stride=2, activation_fn=lrelu,
                              weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                              normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='conv3')
        # e3 is (16 x 16)
        h4 = slim.fully_connected(tf.reshape(h3, [options.batch_size, -1]), 1, activation_fn=None,
                             weights_initializer=fully_init_params, biases_initializer=bias_init_params, scope='lin4')

    return tf.nn.sigmoid(h4), h4