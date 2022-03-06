import logging

import cloudpickle
import pandas as pd
import tensorflow
from keras.layers import TextVectorization
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


class Vectorizer(layers.Layer):
    def __init__(self, vocab_size, sequence_length, batch_size, standardize="lower_and_strip_punctuation", **kwargs):
        super(Vectorizer, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.standardize = standardize
        self.vectorizer = TextVectorization(
            max_tokens=self.vocab_size,
            output_mode="int",
            output_sequence_length=self.sequence_length,
            standardize=self.standardize
        )

    def adapt(self, X):
        self.vectorizer.adapt(X)

    def call(self, X: pd.DataFrame):
        logger.info("Fitting the text vectorizer")
        return self.vectorizer(X)

    def get_config(self):
        config = super(Vectorizer, self).get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "standardize": self.standardize,
        })
        return config

    def save(self, path):
        with open(path, 'wb') as fw:
            cloudpickle.dump(
                {'config': self.vectorizer.get_config(),
                 'weights': self.vectorizer.get_weights(),
                 'vocab': self.vectorizer.get_vocabulary()},
                fw)

    def load(self, path):
        with open(path, 'rb') as fr:
            vec_data = cloudpickle.load(fr)
            self.vectorizer.from_config(vec_data['config'])
            self.vectorizer.set_weights(vec_data['weights'])
            self.vectorizer.set_vocabulary(vec_data['vocab'])
        return self

    @staticmethod
    def output_standardization(input_string):
        return tensorflow.strings.regex_replace(input_string, "[%s]", "")
