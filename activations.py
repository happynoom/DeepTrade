# -*- coding: utf-8 -*-
from __future__ import absolute_import
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects


class ReLU(Layer):
    """Rectified Linear Unit.

    It allows a small gradient when the unit is not active:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        alpha: float >= 0. Negative slope coefficient.

    # References
        - [Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
    """

    def __init__(self, alpha=0.0, max_value=None, **kwargs):
        super(ReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = alpha
        self.max_value = max_value

    def call(self, inputs):
        return K.relu(inputs, alpha=self.alpha, max_value=self.max_value)

    def get_config(self):
        config = {'alpha': self.alpha, 'max_value': self.max_value}
        base_config = super(ReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BiReLU(Layer):
    def __init__(self, alpha=0.0, max_value=None, **kwargs):
        super(BiReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = alpha
        self.max_value = max_value

    def call(self, inputs):
        return K.relu(inputs, alpha=self.alpha, max_value=self.max_value) \
               - K.relu(-inputs, alpha=self.alpha, max_value=self.max_value)

    def get_config(self):
        config = {'alpha': self.alpha, 'max_value': self.max_value}
        base_config = super(BiReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'ReLU': ReLU})
get_custom_objects().update({'BiReLU': BiReLU})
