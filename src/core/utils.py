from pathlib import Path

from core.config import settings


def get_data_path() -> Path:
    return Path(settings.ROOT_PATH_FOR_DYNACONF) / 'data'
