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
import sys
import numpy
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from windpuller import WindPuller, risk_estimation, pairwise_logit
from dataset import DataSet
from feature import extract_from_file
import pickle


def read_ultimate(path, input_shape):
    ultimate_features = numpy.loadtxt(path + "ultimate_feature." + str(input_shape[0]))
    ultimate_features = numpy.reshape(ultimate_features, [-1, input_shape[0], input_shape[1]])
    ultimate_labels = numpy.loadtxt(path + "ultimate_label." + str(input_shape[0]))
    # ultimate_labels = numpy.reshape(ultimate_labels, [-1, 1])
    train_set = DataSet(ultimate_features, ultimate_labels)
    test_features = numpy.loadtxt(path + "ultimate_feature.test." + str(input_shape[0]))
    test_features = numpy.reshape(test_features, [-1, input_shape[0], input_shape[1]])
    test_labels = numpy.loadtxt(path + "ultimate_label.test." + str(input_shape[0]))
    # test_labels = numpy.reshape(test_labels, [-1, 1])
    test_set = DataSet(test_features, test_labels)
    return train_set, test_set


'''
def read_feature(path, input_shape, prefix):
    ultimate_features = numpy.loadtxt("%s/%s_feature.%s" % (path, prefix, str(input_shape[0])))
    ultimate_features = numpy.reshape(ultimate_features, [-1, input_shape[0], input_shape[1]])
    ultimate_labels = numpy.loadtxt("%s/%s_label.%s" % (path, prefix, str(input_shape[0])))
    # ultimate_labels = numpy.reshape(ultimate_labels, [-1, 1])
    train_set = DataSet(ultimate_features, ultimate_labels)
    test_features = numpy.loadtxt("%s/%s_feature.test.%s" % (path, prefix, str(input_shape[0])))
    test_features = numpy.reshape(test_features, [-1, input_shape[0], input_shape[1]])
    test_labels = numpy.loadtxt("%s/%s_label.test.%s" % (path, prefix, str(input_shape[0])))
    # test_labels = numpy.reshape(test_labels, [-1, 1])
    test_set = DataSet(test_features, test_labels)
    return train_set, test_set
'''


def read_feature(path):
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    with open(path, "rb") as fp:
        while True:
            try:
                train_map = pickle.load(fp)
                test_map = pickle.load(fp)
                train_features.extend(train_map["feature"])
                train_labels.extend(train_map["label"])
                test_features.extend(test_map["feature"])
                test_labels.extend(test_map["label"])
                print("read %s successfully. " % train_map["code"])
            except Exception as e:
                break
    return DataSet(numpy.transpose(numpy.asarray(train_features), [0, 2, 1]), numpy.asarray(train_labels)), \
           DataSet(numpy.transpose(numpy.asarray(test_features), [0, 2, 1]), numpy.asarray(test_labels))


def read_separate_feature(path):
    train_sets = {}
    test_sets = {}
    with open(path, "rb") as fp:
        while True:
            try:
                train_map = pickle.load(fp)
                test_map = pickle.load(fp)
                train_sets[train_map["code"]] = DataSet(numpy.transpose(numpy.asarray(train_map["feature"]), [0, 2, 1]),
                                                        numpy.asarray(train_map["label"]))
                test_sets[test_map["code"]] = DataSet(numpy.transpose(numpy.asarray(test_map["feature"]), [0, 2, 1]),
                                                      numpy.asarray(test_map["label"]))
                print("read %s successfully. " % train_map["code"])
            except Exception as e:
                break
    return train_sets, test_sets


def calculate_cumulative_return(labels, pred):
    cr = []
    if len(labels) <= 0:
        return cr
    cr.append(1. * (1. + labels[0] * pred[0]))
    for l in range(1, len(labels)):
        cr.append(cr[l - 1] * (1 + labels[l] * pred[l]))
    for i in range(len(cr)):
        cr[i] = cr[i] - 1
    return cr


def turnover(pred):
    t = 0.0
    for l in range(1, len(pred)):
        t = t + abs(pred[l] - pred[l - 1])
    return t


def evaluate_model(model_path, code, input_shape=[30, 61]):
    extract_from_file("dataset/%s.csv" % code, code)
    train_set, test_set = read_feature("./%s_feature" % code)
    saved_wp = WindPuller.load_model(model_path)
    scores = saved_wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print('Test loss:', scores[0])
    print('test accuracy:', scores[1])
    pred = saved_wp.predict(test_set.images, 1024)
    cr = calculate_cumulative_return(test_set.labels, pred)
    print("changeRate\tpositionAdvice\tprincipal\tcumulativeReturn")
    for i in range(len(test_set.labels)):
        print(str(test_set.labels[i]) + "\t" + str(pred[i]) + "\t" + str(cr[i] + 1.) + "\t" + str(cr[i]))
    print("turnover: %s " % turnover(pred))


def make_model(nb_epochs=100, batch_size=128, lr=0.01, n_layers=1, n_hidden=14, rate_dropout=0.3, loss=risk_estimation):
    train_set, test_set = read_feature("./ultimate_feature")  # read_ultimate("./", input_shape)
    input_shape = [train_set.images.shape[1], train_set.images.shape[2]]
    model_path = 'model.%s' % input_shape[0]
    wp = WindPuller(input_shape=input_shape, lr=lr, n_layers=n_layers, n_hidden=n_hidden, rate_dropout=rate_dropout,
                    loss=loss)
    wp.build_model()
    wp.fit(train_set.images, train_set.labels, batch_size=batch_size,
           nb_epoch=nb_epochs, shuffle=True, verbose=1,
           validation_data=(test_set.images, test_set.labels),
           callbacks=[TensorBoard(histogram_freq=10),
                      ModelCheckpoint(filepath=model_path + '.best.checkpoints', save_best_only=True, mode='min')])
    scores = wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    wp.model.save(model_path)
    saved_wp = wp.load_model(model_path)
    scores = saved_wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print('Test loss:', scores[0])
    print('test accuracy:', scores[1])
    pred = saved_wp.predict(test_set.images, 1024)
    # print(pred)
    # print(test_set.labels)
    pred = numpy.reshape(pred, [-1])
    result = numpy.array([pred, test_set.labels]).transpose()
    with open('output.' + str(input_shape[0]), 'w') as fp:
        for i in range(result.shape[0]):
            for val in result[i]:
                fp.write(str(val) + "\t")
            fp.write('\n')


def make_separate_model(nb_epochs=100, batch_size=128, lr=0.01, n_layers=1, n_hidden=14, rate_dropout=0.3, input_shape=[30, 73]):
    train_sets, test_sets = read_separate_feature("./ultimate_feature")

    wp = WindPuller(input_shape=input_shape, lr=lr, n_layers=n_layers, n_hidden=n_hidden, rate_dropout=rate_dropout)
    wp.build_model()
    for code, train_set in train_sets.items():
        test_set = test_sets[code]
        input_shape = [train_set.images.shape[1], train_set.images.shape[2]]
        print(input_shape)
        model_path = 'model.%s' % code

        print(train_set.images.shape)
        wp.fit(train_set.images, train_set.labels, batch_size=batch_size,
               nb_epoch=nb_epochs, shuffle=False, verbose=1,
               validation_data=(test_set.images, test_set.labels),
               callbacks=[TensorBoard(histogram_freq=1000),
                          ModelCheckpoint(filepath=model_path + '.best.checkpoints', save_best_only=True, mode='min')])
        scores = wp.evaluate(test_set.images, test_set.labels, verbose=0)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        wp.model.save(model_path)
        saved_wp = wp.load_model(model_path)
        scores = saved_wp.evaluate(test_set.images, test_set.labels, verbose=0)
        print('Test loss:', scores[0])
        print('test accuracy:', scores[1])
        pred = saved_wp.predict(test_set.images, 1024)
        # print(pred)
        # print(test_set.labels)
        pred = numpy.reshape(pred, [-1])
        result = numpy.array([pred, test_set.labels]).transpose()
        with open('output.' + str(input_shape[0]), 'w') as fp:
            for i in range(result.shape[0]):
                for val in result[i]:
                    fp.write(str(val) + "\t")
                fp.write('\n')


if __name__ == '__main__':
    operation = "train"
    # input_shape = [30, 102]
    if len(sys.argv) > 1:
        operation = sys.argv[1]
    if operation == "train":
        # make_separate_model(10000, 512, lr=0.0005, n_hidden=14, rate_dropout=0.5, input_shape=[30, 73])
        make_model(30000, 512, lr=0.0004, n_hidden=14, rate_dropout=0.5)
        # make_model(30000, 512, lr=0.01, n_hidden=64, rate_dropout=0.5, loss=pairwise_logit)
    elif operation == "predict":
        evaluate_model("model.30.best.checkpoints", "000001")
    else:
        print("Usage: gossip.py [train | predict]")
