from PIL import Image
import numpy as np
from pathlib import Path

def load_image(path: str) -> Image.Image:
    """
    Load an image from disk and convert to RGB.
    """
    img = Image.open(path).convert("RGB")
    return img

def save_image(image: Image.Image, path: str):
    """
    Save a PIL image to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)

def numpy_to_pil(mask: np.ndarray) -> Image.Image:
    """
    Convert a numpy array (mask) to PIL Image.
    """
    return Image.fromarray(mask.astype(np.uint8))

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to numpy array.
    """
    return np.array(image)
