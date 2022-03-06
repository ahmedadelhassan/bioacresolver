import logging
import logging.config

LOGGER_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
        "json": {
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s filename: %(filename)s funcName: %("
                      "funcName)s ",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
            }
        },
    "handlers": {
        "standard": {
            "class": "logging.StreamHandler",
            "formatter": "json"
        }
    },
    "loggers": {
        "": {
            "handlers": ["standard"],
            "level": logging.INFO
        }
    }
}


def setup_logging():
    logging.config.dictConfig(LOGGER_CONFIG)
