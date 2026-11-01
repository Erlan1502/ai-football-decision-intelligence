import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from api.routers import (
    players_router,
    scouting_router,
    squad_router,
    cv_router,
)

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("moneyball-api")

app = FastAPI(
    title="Football Moneyball 2.0 & Scouting Decision Intelligence API",
    description=(
        "Production-grade sports recruitment decision intelligence platform (2026 Edition). "
        "Integrates PostgreSQL 18 + pgvector HNSW search, LightGBM valuation regressor, "
        "SHAP XAI explainability, Scipy MILP squad optimization, YOLOv11 + ByteTrack CV tracking, "
        "and Gemini 3.8 Flash structured transfer dossiers."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware (allowing Streamlit UI and local clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 Routers
app.include_router(players_router, prefix="/api/v1")
app.include_router(scouting_router, prefix="/api/v1")
app.include_router(squad_router, prefix="/api/v1")
app.include_router(cv_router, prefix="/api/v1")


@app.get("/health", tags=["System"])
def healthcheck():
    """Healthcheck endpoint reporting service status and model configuration."""
    return {
        "status": "ok",
        "service": "Football Moneyball 2.0 API",
        "version": "2.0.0",
        "llm_model": settings.MODEL_NAME,
        "environment": settings.ENVIRONMENT,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"[API Error] Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error_type": type(exc).__name__, "message": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=settings.API_PORT, reload=True)
