from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from ai.config import BASE_DIR
from ai.router_sync import router as sync_router
from ai.router_async import router as async_router
from ai.router_web import router as web_router
from ai.database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # DB 연결이 불가능하면 서버 기동 시 알려준다 (Docker를 먼저 실행).
    init_db()
    yield

app = FastAPI(title="ResNet18 Fine-tuning · FastAPI · Redis · Celery", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.include_router(web_router)
app.include_router(sync_router)
app.include_router(async_router)

@app.get("/health")
def health():
    return {"status": "ok"}
