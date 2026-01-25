# api/routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import shutil, uuid, time
from PIL import Image
from model.model import SegmentationModel
from configs.config import ModelConfig, StorageConfig, DataDirsConfig
from db.database import Database

def get_router(model_config: ModelConfig, data_dirs: DataDirsConfig, storage_config: StorageConfig, database_url=None):
    router = APIRouter()
    db = Database(db_url=database_url)

    # Initialize model once
    model = SegmentationModel(model_config=model_config)

    @router.get("/")
    def health_check():
        return {"status": "API running"}

    @router.post("/segment")
    async def segment_image(file: UploadFile = File(...)):
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_id = str(uuid.uuid4())
        input_path = data_dirs.input_images / f"{image_id}.jpg"
        mask_path = data_dirs.output_masks / f"{image_id}_mask.png"
        overlay_path = data_dirs.overlay_masks / f"{image_id}_overlay.png"

        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if database_url:
            job_id = db.create_job(input_path)

        # Run inference
        start_time = time.time()
        mask, color_mask_img, overlay_img = model.predict(str(input_path))
        inference_time = int((time.time() - start_time) * 1000)

        # Save colored and overlay mask
        color_mask_img.save(mask_path)
        overlay_img.save(overlay_path)

        # store metadata to DB using database_url
        if database_url:
            db.complete_job(
                job_id,
                mask_path,
                overlay_path,
                inference_time
            )

        return {
            "image_id": image_id,
            "input_image": str(input_path),
            "mask_image": str(mask_path),
            "overlay_image": str(mask_path),
            "model_name": model_config.name,
            "inference_time_ms": inference_time
        }

    @router.get("/mask/{image_id}")
    def get_mask(image_id: str):
        mask_file = data_dirs.output_masks / f"{image_id}_mask.png"
        if not mask_file.exists():
            raise HTTPException(status_code=404, detail="Mask not found")
        return FileResponse(path=mask_file, media_type="image/png")

    return router
