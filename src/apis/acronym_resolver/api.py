from fastapi import APIRouter

from apis.acronym_resolver.routes.v1 import acronym_resolver as acronym_resolver_v1

router = APIRouter()
router.include_router(acronym_resolver_v1.router, prefix='/acronyms/v1', tags=["acronym resolver"])
