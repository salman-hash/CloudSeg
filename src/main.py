# src/main.py
import json
from pathlib import Path
import os
from dotenv import load_dotenv

from fastapi import FastAPI
from api.routes import get_router  # API router function that accepts config objects
from configs.config import ModelConfig, APIConfig, StorageConfig, DataDirsConfig


load_dotenv()  # optional .env
DATABASE_URL = os.getenv("DATABASE_URL")  # fallback if not set


BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR / "testConfig.json") as f:
    config_json = json.load(f)

MODEL = ModelConfig(config_json.get("model", {}))
API = APIConfig(config_json.get("api", {}))
STORAGE = StorageConfig(config_json.get("storage", {}))
DATA_DIRS = DataDirsConfig(config_json.get("data_dirs", {}))


app = FastAPI(title="Cloud Segmentation API", version="1.0.0")

# Include API router with config objects, We pass the configs + DATABASE_URL to router
app.include_router(
    get_router(model_config=MODEL,
               data_dirs=DATA_DIRS,
               storage_config=STORAGE,
               database_url=DATABASE_URL),
    prefix="/api"
)


@app.get("/")
def root():
    return {"status": "Cloud Segmentation API running"}

# =========================
# Entry point for uvicorn
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API.host, port=API.port)
