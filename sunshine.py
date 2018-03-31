# -*- coding: utf-8 -*-
# Copyright 2017 The Xiaoyu Fang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from tensorflow.contrib import rnn
import os

from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.python.ops.init_ops import glorot_uniform_initializer, orthogonal_initializer
from rawdata import RawData, read_sample_data
from dataset import DataSet
from chart import extract_feature
import numpy
from tensorflow.contrib.layers.python.layers.layers import batch_norm
import sys
from numpy.random import seed


class SunShine(object):

    def __init__(self, step, input_size, starter_learning_rate, hidden_size, nclasses, decay_step=500, decay_rate=1.0, cost=0.0002):
        '''
        Initialize parameters for the SmartTrader
        :param step: time steps of the feature
        :param input_size: size of each time step of the feature
        :param starter_learning_rate: initial learning rate, the learning rate decays along global train step
        :param hidden_size: hidden units of the LSTM layer
        :param nclasses: number of classes, should be 1
        :param decay_step: learning rate decay step
        :param decay_rate: learning rate decay rate
        :param cost: the constant cost for money occupied by buying stock
        '''
        self.step = step
        self.input_size = input_size
        self.global_step = None
        self.starter_learning_rate = starter_learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.learning_rate = None
        self.hidden_size = hidden_size
        self.nclasses = nclasses
        self.position = None
        self.summary_op = None
        self.weights = None
        self.biases = None
        self.cost = cost
        self.loss = None
        self.avg_position = None
        self.keep_rate = None
        self.x = None
        self.y = None
        self.is_training = None

    def _create_learning_rate(self):
        '''
        create learning rate
        :return:
        '''
        with tf.variable_scope("parameter"):
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                   self.decay_step, self.decay_rate, staircase=True, name="learning_rate")

    def _create_placeholders(self):
        with tf.variable_scope("input"):
            self.x = tf.placeholder(tf.float32, shape=[None, self.nclasses, self.step, self.input_size], name='history_feature')
            self.y = tf.placeholder(tf.float32, shape=[None, self.nclasses], name='target')
            self.is_training = tf.placeholder(tf.bool, name='mode')
            self.keep_rate = tf.placeholder(tf.float32, name='keep_rate')

    def _create_weights(self):
        with tf.variable_scope("weights"):
            self.weights = {
                'out': tf.get_variable("weights", [self.hidden_size, self.nclasses],
                                       initializer=tf.random_normal_initializer(mean=0, stddev=0.01, seed=1))
            }
            self.biases = {
                'out': tf.get_variable("bias", [self.nclasses], initializer=tf.random_normal_initializer(mean=0, stddev=0.01, seed=1))
            }

    def batch_norm_layer(self, signal, scope):
        '''
        batch normalization layer before activation
        :param signal: input signal
        :param scope: name scope
        :return: normalized signal
        '''
        # Note: is_training is tf.placeholder(tf.bool) type
        return tf.cond(self.is_training,
                       lambda: batch_norm(signal, is_training=True, param_initializers={"beta": tf.constant_initializer(3.), "gamma": tf.constant_initializer(2.5)},
                                          center=True, scale=True, activation_fn=tf.nn.relu, decay=1., scope=scope),
                       lambda: batch_norm(signal, is_training=False, param_initializers={"beta": tf.constant_initializer(3.), "gamma": tf.constant_initializer(2.5)},
                                          center=True, scale=True, activation_fn=tf.nn.relu, decay=1.,
                                          scope=scope, reuse=True))

    def _create_loss(self):
        '''
        Risk estimation loss function. The output is the planed position we should hold to next day. The change rate of
        next day is self.y, so we loss two categories of money: - self.y * self.position is trade loss,
        cost * self.position is constant loss because of tax and like missing profit of buying national debt. Therefore,
        the loss function is formulated as: 100 * (- self.y * self.position + cost * self.position) = -100 * ((self.y - cost) * self.position)
        :return:
        '''
        #with tf.device("/cpu:0"):
        xx = tf.unstack(self.x, self.step, 1)
        lstm_cell = rnn.LSTMCell(self.hidden_size, forget_bias=1.0, initializer=orthogonal_initializer())
        dropout_cell = DropoutWrapper(lstm_cell, input_keep_prob=self.keep_rate, output_keep_prob=self.keep_rate, state_keep_prob=self.keep_rate)
        outputs, states = rnn.static_rnn(dropout_cell, xx, dtype=tf.float32)
        signal = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
        scope = "activation_batch_norm"
        norm_signal = self.batch_norm_layer(signal, scope=scope)
        # batch_norm(signal, 0.9, center=True, scale=True, epsilon=0.001, activation_fn=tf.nn.relu6,
        #           is_training=is_training, scope="activation_batch_norm", reuse=False)
        self.position = tf.nn.relu6(norm_signal, name="relu_limit") / 6.
        self.avg_position = tf.reduce_mean(self.position)
        # self.cost = 0.0002
        self.loss = -100. * tf.reduce_mean(tf.multiply((self.y - self.cost), self.position, name="estimated_risk"))

    def _create_optimizer(self):
        '''
        create optimizer
        :return:
        '''
        #with tf.device("/cpu:0"):
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, name="optimizer").minimize(self.loss, global_step=self.global_step)

    def _create_summary(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.histogram("histogram loss", self.loss)
        tf.summary.scalar("average position", self.avg_position)
        tf.summary.histogram("histogram position", self.avg_position)
        self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_learning_rate()
        self._create_placeholders()
        self._create_weights()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()


if __name__ == '__main__':
    pass
