import logging
from typing import Dict

from fastapi import FastAPI
from fastapi_utils.timing import add_timing_middleware

from apis.acronym_resolver.api import router as api_resolver
from core.config import settings
from core.logging import setup_logging


def create_app():
    setup_logging()

    app = FastAPI(title="Biomedical Acronym Resolver", version=settings.api.version,
                  description="""A simple ML model to disambiguate medical acronyms""")

    @app.get('/')
    def root() -> Dict[str, str]:
        return {"message": "Welcome to the Biomedical Acronym Resolver"}

    app.include_router(api_resolver, prefix='/api')

    timing_logger = logging.getLogger('timing')
    add_timing_middleware(app, record=timing_logger.info, prefix="app", exclude="untimed")

    return app
