import logging

import pandas as pd
from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get('/resolve')
def acronym_resolver(input_string: str, request: Request) -> JSONResponse:
    """
    An endpoint that takes a string that can include acronyms, and returns another string with acronyms resolved
    """

    try:
        pipeline = request.app.state.pipeline
        resolved_string = pipeline.predict_one(input_string)
        return JSONResponse(status_code=status.HTTP_200_OK, content=resolved_string)

    except Exception as err:
        logger.exception("Unable to resolve string",
                         extra={"input_string": input_string})
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=input_string)
