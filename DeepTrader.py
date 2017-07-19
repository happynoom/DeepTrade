import numpy
from tensorflow.contrib.keras.python.keras.layers import Dense, LSTM, Activation, BatchNormalization, initializers, Dropout
from tensorflow.contrib.keras.python.keras.models import Sequential, load_model
from tensorflow.contrib.keras.python.keras.optimizers import RMSprop
from tensorflow.contrib.keras.python.keras.initializers import Constant
from tensorflow.contrib.keras.python.keras.callbacks import TensorBoard, ModelCheckpoint
from dataset import DataSet
from smarttrader import read_sample_data
from chart import extract_feature
from tensorflow.contrib.keras.python.keras import backend as K


def relu_limited(x, alpha=0., max_value=1.):
    return K.relu(x, alpha=alpha, max_value=max_value)


def risk_estimation(y_true, y_pred):
    return -100. * K.mean((y_true - 0.0002) * y_pred)


class WindPuller(object):
    def __init__(self, input_shape, lr=0.01, n_layers=2, n_hidden=8, rate_dropout=0.2, loss=risk_estimation):
        print("initializing..., learing rate %s, n_layers %s, n_hidden %s, dropout rate %s." % (
        lr, n_layers, n_hidden, rate_dropout))
        self.model = Sequential()
        self.model.add(Dropout(rate=rate_dropout, input_shape=(input_shape[0], input_shape[1])))
        for i in range(0, n_layers - 1):
            self.model.add(LSTM(n_hidden * 4, return_sequences=True, activation='tanh',
                                recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                                recurrent_initializer='orthogonal', bias_initializer='zeros',
                                dropout=rate_dropout, recurrent_dropout=rate_dropout))
        self.model.add(LSTM(n_hidden, return_sequences=False, activation='tanh',
                            recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                            recurrent_initializer='orthogonal', bias_initializer='zeros',
                            dropout=rate_dropout, recurrent_dropout=rate_dropout))
        self.model.add(Dense(1, kernel_initializer=initializers.glorot_uniform()))
        # self.model.add(BatchNormalization(axis=-1, moving_mean_initializer=Constant(value=0.5),
        #               moving_variance_initializer=Constant(value=0.25)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation("relu(alpha=0., max_value=1.0)"))
        opt = RMSprop(lr=lr)
        self.model.compile(loss=loss,
                           optimizer=opt,
                           metrics=['accuracy'])

    def fit(self, x, y, batch_size=32, nb_epoch=100, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0):
        self.model.fit(x, y, batch_size, nb_epoch, verbose, callbacks,
                       validation_split, validation_data, shuffle, class_weight, sample_weight,
                       initial_epoch)

    def save(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)
        return self

    def evaluate(self, x, y, batch_size=32, verbose=1,
                 sample_weight=None, **kwargs):
        return self.model.evaluate(x, y, batch_size, verbose,
                                   sample_weight)

    def predict(self, x, batch_size=32, verbose=0):
        return self.model.predict(x, batch_size, verbose)


def make_model(input_shape, train_set, test_set, nb_epochs=100, batch_size=128, lr=0.01, n_layers=1, n_hidden=16, rate_dropout=0.3):
    model_path = 'model.%s' % input_shape[0]
    wp = WindPuller(input_shape=input_shape, lr=lr, n_layers=n_layers, n_hidden=n_hidden, rate_dropout=rate_dropout)
    # train_set, test_set = read_feature("./ultimate_feature.%s" % input_shape[0])  # read_ultimate("./", input_shape)
    wp.fit(train_set.images, train_set.labels, batch_size=batch_size,
           nb_epoch=nb_epochs, shuffle=True, verbose=1,
           validation_data=(test_set.images, test_set.labels),
           callbacks=[TensorBoard(histogram_freq=1),
                      ModelCheckpoint(filepath=model_path + '.best', save_best_only=True, mode='min')])
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


def main(operation='train'):
    selector = ["ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME"]
    input_shape = [30, 61]  # [length of time series, length of feature]
    raw_data = read_sample_data("toy_stock.csv")
    moving_features, moving_labels = extract_feature(raw_data=raw_data, selector=selector, window=input_shape[0],
                                                     with_label=True, flatten=False)
    moving_features = numpy.asarray(moving_features)
    moving_features = numpy.transpose(moving_features, [0, 2, 1])
    moving_labels = numpy.asarray(moving_labels)
    moving_labels = numpy.reshape(moving_labels, [moving_labels.shape[0], 1])
    validation_size = 600
    train_set = DataSet(moving_features[:-validation_size], moving_labels[:-validation_size])
    test_set = DataSet(moving_features[-validation_size:], moving_labels[-validation_size:])
    if operation == 'train':
        make_model(input_shape, train_set, test_set)


if __name__ == '__main__':
    main("train")
