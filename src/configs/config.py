import json
import os
from pathlib import Path

class ModelConfig:
    def __init__(self, cfg: dict):
        self.name = cfg.get("name", "deeplabv3_resnet50")
        self.device = cfg.get("device", "cpu")
        self.input_size = tuple(cfg.get("input_size", [512, 512]))


class APIConfig:
    def __init__(self, cfg: dict):
        self.host = cfg.get("host", "0.0.0.0")
        self.port = cfg.get("port", 5000)


class StorageConfig:
    def __init__(self, cfg: dict):
        self.azure_input_container = cfg.get("azure_input_container")
        self.azure_output_container = cfg.get("azure_output_container")


class DataDirsConfig:
    def __init__(self, cfg: dict):
        self.base_dir = Path(cfg.get("base_dir", "../"))
        self.input_images = Path(cfg.get("input_images", "../data/input_images"))
        self.output_masks = Path(cfg.get("output_masks", "../data/output_masks"))

        # Ensure directories exist
        self.input_images.mkdir(parents=True, exist_ok=True)
        self.output_masks.mkdir(parents=True, exist_ok=True)

