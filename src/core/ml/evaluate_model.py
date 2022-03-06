import logging

from core.logging import setup_logging
from core.ml.pipeline.pipeline import Pipeline
from core.data.prepare_dataset import get_test_data

logger = logging.getLogger('evaluate_model')
setup_logging()


def evaluate_model():
    test_dataset = get_test_data()
    pipeline = Pipeline().load()
    accuracy = pipeline.evaluate(test_dataset['input_sentence'], test_dataset['output_sentence'])
    logger.info(f"Average accuracy is: {accuracy}")


if __name__ == '__main__':
    logger.info("Evaluating model")
    evaluate_model()
