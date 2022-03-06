import logging

import pandas as pd

from core.config import settings
from core.logging import setup_logging
from core.utils import get_data_path
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
setup_logging()


def _load_train_dataset() -> pd.DataFrame:
    dataset_path = get_data_path() / 'training' / settings.data.training.dataset
    df = pd.read_csv(dataset_path, sep='|')
    return df[df['labels'].notnull()]


def _load_test_dataset() -> pd.DataFrame:
    dataset_path = get_data_path() / 'test' / settings.data.test.dataset
    df = pd.read_csv(dataset_path, sep='|')
    return df[df['labels'].notnull()]


def _preprocess_sequences(dataset) -> pd.DataFrame:
    # Add start and end of sequence to output sentences
    dataset['output_sentence'] = dataset['output_sentence'].apply(
        lambda s: f'[start] {s} [end]'
    )
    return dataset


def split_train_val_data(dataset: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    logger.info("Stratified splitting for training and validation sets")
    dataset_train, dataset_val = train_test_split(dataset,
                                                  test_size=settings.pipeline.preprocessing.val_split_percentage,
                                                  random_state=2,
                                                  shuffle=True,
                                                  stratify=dataset['labels'])
    logger.info(f"Train set size: {len(dataset_train)}")
    logger.info(f"Validation set size: {len(dataset_val)}")
    return dataset_train, dataset_val


def get_train_data() -> pd.DataFrame:
    full_dataset = _load_train_dataset()
    dataset_train = _preprocess_sequences(full_dataset)
    logger.info(f"Train set size: {len(dataset_train)}")
    return dataset_train


def get_test_data() -> pd.DataFrame:
    test_dataset = _load_test_dataset()
    dataset_test = _preprocess_sequences(test_dataset)
    logger.info(f"Test set size: {len(dataset_test)}")
    return dataset_test
