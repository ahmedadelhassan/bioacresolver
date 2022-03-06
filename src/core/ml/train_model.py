import logging

from mlflow import mlflow

from core.config import settings
from core.data.prepare_dataset import get_train_data
from core.logging import setup_logging
from core.ml.pipeline.pipeline import Pipeline

logger = logging.getLogger('train_model')
setup_logging()


def train_model():
    logger.info("Setting MLflow tracking")
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    experiment = mlflow.get_experiment('acronym_disambiguation')
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        mlflow.tensorflow.autolog()

        train_data = get_train_data()
        pipeline = Pipeline()
        logger.info("Starting model training")
        trained_model = pipeline.fit(train_data, validate=False)
        accuracy = trained_model.cls.history.history['accuracy'][-1]
        val_accuracy = trained_model.cls.history.history['val_accuracy'][-1]
        logger.info(f"Average training accuracy is: {accuracy}, Average validation accuracy is: {val_accuracy}")
        logger.info(f"Saving model artifacts")
        trained_model.save()


if __name__ == '__main__':
    logger.info(f"Training model")
    train_model()
