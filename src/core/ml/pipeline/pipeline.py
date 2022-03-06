import logging

import pandas as pd
import tensorflow

from core.config import settings
from core.logging import setup_logging
from core.ml.pipeline.transformer import Transformer
from core.ml.pipeline.vectorizer import Vectorizer
from core.ml.prepare_dataset import get_train_data, split_train_val_data
from core.utils import get_data_path

logger = logging.getLogger(__name__)
setup_logging()


class Pipeline:
    def __init__(self):
        logger.info("Initializing model pipeline")
        self.params = {**settings.pipeline.transformer, **settings.pipeline.vectorizer}
        self.checkpoint_path = get_data_path() / 'model_weights' / self.params['checkpoint_path']

        self.train_data = None
        self.val_data = None

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
        input_vec = self.input_vectorizer.fit(input)
        output_vec = self.output_vectorizer.fit(output)
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
            self.train_data, self.val_data = split_train_val_data(X)
            X_val = self.make_dataset(self.val_data)
        else:
            self.train_data = X
            X_val = self.make_dataset(self.train_data)

        X_train = self.make_dataset(self.train_data)

        cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                                 verbose=1)

        self.cls.fit(X_train, epochs=self.params['epochs'], validation_data=X_val, callbacks=[cp_callback])

# TODO: save the whole pipeline not only the TF model
