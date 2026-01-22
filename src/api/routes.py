# api/routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import shutil, uuid, time
from PIL import Image
from model.model import SegmentationModel

def get_router(model_config, data_dirs, storage_config, database_url=None):
    router = APIRouter()

    # Initialize model once
    model = SegmentationModel(device=model_config.device)

    @router.get("/")
    def health_check():
        return {"status": "API running"}

    @router.post("/segment")
    async def segment_image(file: UploadFile = File(...)):
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_id = str(uuid.uuid4())
        input_path = data_dirs.input_images / f"{image_id}.jpg"
        output_path = data_dirs.output_masks / f"{image_id}_mask.png"

        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run inference
        start_time = time.time()
        mask = model.predict(str(input_path))
        inference_time = int((time.time() - start_time) * 1000)

        # Save mask
        mask_image = Image.fromarray(mask)
        mask_image.save(output_path)

        # todo: store metadata to DB using database_url
        # if database_url:
        #     ...

        return {
            "image_id": image_id,
            "input_image": str(input_path),
            "mask_image": str(output_path),
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
