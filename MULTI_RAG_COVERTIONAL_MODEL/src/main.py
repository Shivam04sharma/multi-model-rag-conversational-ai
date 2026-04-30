"""Multi RAG Conversational AI Model — FastAPI entry point."""

import socket
from contextlib import asynccontextmanager

import structlog
from config import settings
from db.session import close_db, init_db, init_redis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routes import rag, voice

try:
    from core.middleware import add_observability_middleware

    _HAS_OBSERVABILITY = True
except ImportError:
    _HAS_OBSERVABILITY = False

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await init_redis()

    logger.info(
        "startup",
        service=settings.service_name,
        env=settings.env,
        port=settings.port,
    )
    yield

    await close_db()
    logger.info("shutdown", service=settings.service_name)


app = FastAPI(
    title="Multi RAG Conversational AI Model",
    version="1.0.0",
    lifespan=lifespan,
    root_path="",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if _HAS_OBSERVABILITY:
    add_observability_middleware(app)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "message": "An unexpected error occurred."},
    )


app.include_router(rag.router, prefix="/api/v1/rag", tags=["rag"])
app.include_router(voice.router, prefix="/api/v1/voice", tags=["voice"])


@app.get("/health", tags=["health"])
@app.get("/actuator/health", tags=["health"])
async def health():
    return {
        "status": "ok",
        "service": settings.service_name,
        "version": app.version,
        "env": settings.env,
        "app_instance": socket.gethostname(),
    }
