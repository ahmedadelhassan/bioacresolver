import logging
import re

import pandas as pd

from core.config import settings
from core.logging import setup_logging
from core.utils import get_data_path

logger = logging.getLogger('create_datasets')
setup_logging()


def _remove_nonalphnumeric(s: str) -> str:
    pattern = re.compile(r'[^A-Za-z0-9 ]+')
    return pattern.sub(' ', s).lower()


def _load_acronyms_mapping() -> dict:
    """ Load acronyms mapping """
    mapping_path = get_data_path() / settings.data.mapping.mapping_data
    df = pd.read_csv(mapping_path, sep='|')
    df = df[settings.data.mapping.mapping_columns].drop_duplicates()
    df['expansion'] = df.apply(lambda s: s['expansion'].lower(), axis=1)
    return df.set_index('expansion') \
        .to_dict(orient='index')


def _load_test_samples() -> pd.DataFrame:
    """ Load toy test data """
    raw_test_data_path = get_data_path() / settings.data.raw.raw_test_data
    raw_df = pd.read_csv(raw_test_data_path, sep='|')
    raw_df['input_sentence'], raw_df['labels'] = raw_df['sample'], raw_df['expansion'].str.lower()
    raw_df['output_sentence'] = raw_df. \
        apply(lambda s: s['input_sentence'].replace(s['acronym'], s['expansion']),
              axis=1)
    return raw_df[['input_sentence', 'output_sentence', 'labels']]


def _load_raw_sentence_data() -> pd.DataFrame:
    """ Load raw sentence data """
    raw_sentence_data_path = get_data_path() / settings.data.raw.raw_sentence_data
    with open(raw_sentence_data_path, 'r') as fr:
        sentences = [line.replace('\n', '') for line in fr.readlines()]
    return pd.DataFrame(sentences, columns=['output_sentence'])


def create_dataset() -> None:
    """ Create training and test dataset from raw sentences by replacing the expanded words with their
    acronyms """
    logger.info("Creating training data by mapping expanded words to acronyms")
    acronyms_mapping = _load_acronyms_mapping()
    train_sentences = _load_raw_sentence_data()

    def map_acronyms(s: str, mapping: dict):
        acronyms = []
        for key, value in mapping.items():
            if (key in s) or (_remove_nonalphnumeric(key) in s):
                acronyms.append(key)
            s = s.replace(key, value['acronym'])
        return s, ', '.join(acronyms)

    train_sentences['input_sentence'], train_sentences['labels'] = zip(*train_sentences.apply(
        lambda s: map_acronyms(s['output_sentence'], acronyms_mapping),
        axis=1))
    train_data_path = get_data_path() / settings.data.training.dataset
    train_sentences.to_csv(train_data_path, sep='|', index=False)

    logger.info("Creating test data by mapping acronyms to expanded forms")
    test_sentences = _load_test_samples()
    test_data_path = get_data_path() / settings.data.test.dataset
    test_sentences.to_csv(test_data_path, sep='|', index=False)


if __name__ == '__main__':
    logger.info("Creating training dataset from raw input")
    create_dataset()
