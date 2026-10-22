import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agents.registry import sync_agent_registry
from app.api.routes import router as api_router
from app.config import settings
from app.data.seed_data import ensure_seeded
from app.database import Base, SessionLocal, engine

ACCESS_CODE = "thesisout!"


async def _background_seed() -> None:
    db = SessionLocal()
    try:
        await ensure_seeded(db)
    except Exception:
        pass
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        sync_agent_registry(db)
    finally:
        db.close()
    seed_task = asyncio.create_task(_background_seed())
    yield
    seed_task.cancel()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def enforce_access_code_and_bot_shield(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api"):
        if path == "/api/auth/verify":
            return await call_next(request)
        header_code = request.headers.get("X-Access-Code", "")
        query_code = request.query_params.get("access_code", "")
        cookie_code = request.cookies.get("thesisout_access", "")
        if header_code != ACCESS_CODE and query_code != ACCESS_CODE and cookie_code != ACCESS_CODE:
            return JSONResponse(
                status_code=401,
                content={"detail": "Access code required."},
            )
    return await call_next(request)


@app.post("/api/auth/verify")
async def verify_access_code(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    code = str(body.get("code", "")).strip()
    honeypot = str(body.get("website_url", "")).strip()
    if honeypot:
        return JSONResponse(status_code=403, content={"valid": False, "detail": "Bot detected."})
    if code == ACCESS_CODE:
        resp = JSONResponse(content={"valid": True})
        resp.set_cookie(
            key="thesisout_access",
            value=ACCESS_CODE,
            max_age=60 * 60 * 24 * 30,
            httponly=False,
            samesite="lax",
        )
        return resp
    return JSONResponse(status_code=401, content={"valid": False, "detail": "Invalid access code."})


app.include_router(api_router, prefix="/api")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "primary_llm": settings.PRIMARY_LLM_MODEL,
        "fallback_llm": settings.FALLBACK_LLM_MODEL,
    }


# Serve compiled React SPA in Cloud Run / Production
FRONTEND_DIST = settings.BASE_DIR / "frontend" / "dist"
if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(index_file))
    return {"message": "ThesisOut API is running. Build frontend/dist for SPA serving."}
