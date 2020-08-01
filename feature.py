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

import os
from rawdata import RawData, read_sample_data
from dataset import DataSet
from chart import extract_feature
import numpy
import pickle


def extract_from_file(filepath, output_prefix, window=30):
    days_for_test = 700
    # input_shape = [30, 61]  # [length of time series, length of feature]
      # input_shape[0]
    fp = open("%s_feature" % output_prefix, "w")
    # lp = open("%s_label.%s" % (output_prefix, window), "w")
    # fpt = open("%s_feature.test.%s" % (output_prefix, window), "w")
    # lpt = open("%s_label.test.%s" % (output_prefix, window), "w")

    selector = ["ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME", "CROSS_PRICE"]

    raw_data = read_sample_data(filepath)
    moving_features, moving_labels = extract_feature(raw_data=raw_data, selector=selector, window=window,
                                                     with_label=True, flatten=False)
    print("feature extraction done, start writing to file...")
    train_end_test_begin = moving_features.shape[0] - days_for_test
    if train_end_test_begin < 0:
        train_end_test_begin = 0
    train_set = {"code": output_prefix, "feature": moving_features[:train_end_test_begin],
                 "label": moving_labels[:train_end_test_begin]}
    pickle.dump(train_set, fp, 2)
    test_set = {"code": output_prefix, "feature": moving_features[train_end_test_begin:],
                "label": moving_labels[train_end_test_begin:]}
    pickle.dump(test_set, fp, 2)
    '''
    for i in range(0, train_end_test_begin):
        for item in moving_features[i]:
            fp.write("%s\t" % item)
        fp.write("\n")
    for i in range(0, train_end_test_begin):
        lp.write("%s\n" % moving_labels[i])
    # test set
    for i in range(train_end_test_begin, moving_features.shape[0]):
        for item in moving_features[i]:
            fpt.write("%s\t" % item)
        fpt.write("\n")
    for i in range(train_end_test_begin, moving_features.shape[0]):
        lpt.write("%s\n" % moving_labels[i])
    '''
    fp.close()
    # lp.close()
    # fpt.close()
    # lpt.close()


if __name__ == '__main__':
    days_for_test = 700
    # input_shape = [30, 61]  # [length of time series, length of feature]
    window = 30  # input_shape[0]
    fp = open("ultimate_feature", "wb")
    # lp = open("ultimate_label.%s" % window, "wb")
    # fpt = open("ultimate_feature.test.%s" % window, "wb")
    # lpt = open("ultimate_label.test.%s" % window, "wb")

    selector = ["ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME", "CROSS_PRICE"]
    dataset_dir = "./dataset"
    for filename in os.listdir(dataset_dir):
        #if filename != '000001.csv':
        #    continue
        print("processing file: " + filename)
        filepath = dataset_dir + "/" + filename
        raw_data = read_sample_data(filepath)
        moving_features, moving_labels = extract_feature(raw_data=raw_data, selector=selector, window=window,
                                                         with_label=True, flatten=False)
        print("feature extraction done, start writing to file...")
        train_end_test_begin = moving_features.shape[0] - days_for_test
        if train_end_test_begin < 0:
            train_end_test_begin = 0

        train_set = {"code": filename[:filename.rfind(".")], "feature": moving_features[:train_end_test_begin],
                     "label": moving_labels[:train_end_test_begin]}
        pickle.dump(train_set, fp, 2)
        test_set = {"code": filename[:filename.rfind(".")], "feature": moving_features[train_end_test_begin:],
                     "label": moving_labels[train_end_test_begin:]}
        pickle.dump(test_set, fp, 2)
        '''
        for i in range(0, train_end_test_begin):
            for item in moving_features[i]:
                fp.write("%s\t" % item)
            fp.write("\n")
        for i in range(0, train_end_test_begin):
            lp.write("%s\n" % moving_labels[i])
        # test set
        for i in range(train_end_test_begin, moving_features.shape[0]):
            for item in moving_features[i]:
                fpt.write("%s\t" % item)
            fpt.write("\n")
        for i in range(train_end_test_begin, moving_features.shape[0]):
            lpt.write("%s\n" % moving_labels[i])
        '''
    fp.close()
    # lp.close()
    # fpt.close()
    # lpt.close()
