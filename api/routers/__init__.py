from api.routers.players import router as players_router
from api.routers.scouting import router as scouting_router
from api.routers.squad import router as squad_router
from api.routers.cv import router as cv_router

__all__ = [
    "players_router",
    "scouting_router",
    "squad_router",
    "cv_router",
]
