import logging

import pandas as pd
import tensorflow
from keras.layers import TextVectorization

from core.logging import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


class Vectorizer:
    def __init__(self, vocab_size, sequence_length, batch_size, standardize="lower_and_strip_punctuation"):
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

    def fit(self, X: pd.DataFrame, standardize=None):
        logger.info("Fitting the text vectorizer")
        return self.vectorizer(X)

    def transform(self):
        pass

    def inverse_transform(self):
        pass

    @staticmethod
    def output_standardization(input_string):
        return tensorflow.strings.regex_replace(input_string, "[%s]", "")
