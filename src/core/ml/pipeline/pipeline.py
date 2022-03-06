import logging

import numpy as np
import pandas as pd
import tensorflow
from tensorflow.python.keras.metrics import Accuracy

from core.config import settings
from core.ml.pipeline.transformer import Transformer
from core.ml.pipeline.vectorizer import Vectorizer
from core.data.prepare_dataset import split_train_val_data
from core.utils import get_data_path

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self):
        logger.info("Initializing model pipeline")
        self.params = {**settings.pipeline.transformer, **settings.pipeline.vectorizer, **settings.pipeline}
        self.pipeline_artifacts_path = get_data_path() / self.params['pipeline_artifacts_path']

        self.input_vectorizer = Vectorizer(
            vocab_size=self.params['vocab_size'],
            sequence_length=self.params['sequence_length'],
            batch_size=self.params['batch_size'],
        )
        self.output_vectorizer = Vectorizer(
            vocab_size=self.params['vocab_size'],
            sequence_length=self.params['sequence_length'] + 1,
            batch_size=self.params['batch_size'],
            standardize=Vectorizer.output_standardization
        )

        self.cls = Transformer(
            sequence_length=self.params['sequence_length'],
            vocab_size=self.params['vocab_size'],
            embedding_dim=self.params['embedding_dim'],
            latent_dim=self.params['latent_dim'],
            num_heads=self.params['num_heads'],
        )()

    def _format_dataset(self, input, output):
        input_vec = self.input_vectorizer(input)
        output_vec = self.output_vectorizer(output)
        return ({"encoder_inputs": input_vec, "decoder_inputs": output_vec[:, :-1], }, output_vec[:, 1:])

    def make_dataset(self, dataset: pd.DataFrame):
        input_sentence, output_sentence = dataset['input_sentence'].to_list(), dataset['output_sentence'].to_list()
        self.input_vectorizer.adapt(input_sentence)
        self.output_vectorizer.adapt(output_sentence)
        dataset = tensorflow.data.Dataset.from_tensor_slices((input_sentence, output_sentence))
        dataset = dataset.batch(self.params['batch_size'])
        dataset = dataset.map(self._format_dataset)
        return dataset.shuffle(2048).prefetch(16).cache()

    def fit(self, X: pd.DataFrame, validate=False):
        logger.info("Fitting and saving model pipeline")
        if validate:
            train_data, val_data = split_train_val_data(X)
            X_val = self.make_dataset(val_data)
        else:
            train_data = X
            X_val = self.make_dataset(train_data)

        X_train = self.make_dataset(train_data)

        self.cls.fit(X_train, epochs=self.params['epochs'], validation_data=X_val)

        return self

    def predict_one(self, input_sentence):
        output_vocab = self.output_vectorizer.vectorizer.get_vocabulary()
        output_index_lookup = dict(zip(range(len(output_vocab)), output_vocab))
        tokenized_input_sentence = self.input_vectorizer.vectorizer([input_sentence])
        decoded_sentence = "[start]"
        for i in range(self.params['decoded_sequence_length']):
            tokenized_target_sentence = self.output_vectorizer.vectorizer([decoded_sentence])[:, :-1]
            predictions = self.cls([tokenized_input_sentence, tokenized_target_sentence])

            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = output_index_lookup[sampled_token_index]
            decoded_sentence += " " + sampled_token

            if sampled_token == "[end]":
                break
        return decoded_sentence.replace('[start] ', '').replace(' [end]', '')

    def predict(self, X: pd.DataFrame):
        return X.apply(lambda s: self.predict_one(s))

    def evaluate(self, X, y, metric=Accuracy):
        m = metric()
        y_pred = self.predict(X)
        y_pred = self.output_vectorizer.vectorizer(y_pred)
        y_true = self.output_vectorizer.vectorizer(y)
        m.update_state(y_pred=y_pred.numpy(), y_true=y_true.numpy())
        return m.result()

    def save(self, path=None):
        artifacts_path = path or self.pipeline_artifacts_path
        # save each step separately (workaround for not being picklable)
        self.input_vectorizer.save(get_data_path() / self.params['input_vectorizer_path'])
        self.output_vectorizer.save(get_data_path() / self.params['output_vectorizer_path'])
        self.input_vectorizer = None
        self.output_vectorizer = None
        self.cls.save_weights(artifacts_path / self.params['model_weights'])

    def load(self, artifacts_path=None):
        artifacts_path = artifacts_path or self.pipeline_artifacts_path
        self.input_vectorizer.load(get_data_path() / self.params['input_vectorizer_path'])
        self.output_vectorizer.load(get_data_path() / self.params['output_vectorizer_path'])
        self.cls.load_weights(artifacts_path / self.params['model_weights'])
        return self
