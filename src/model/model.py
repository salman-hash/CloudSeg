# src/model/model.py
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

class SegmentationModel:
    def __init__(self, model_config):
        """
        Initialize model using a ModelConfig object
        """
        self.device = model_config.device
        self.model_name = model_config.name
        self.input_size = model_config.input_size

        # Load pretrained DeepLabV3
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

        # Preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_color_mask(self, mask: np.ndarray) -> Image.Image:
        """Convert class mask to color mask"""
        # Generate a consistent color palette
        np.random.seed(42)
        palette = np.random.randint(0, 256, size=(self.num_classes, 3), dtype=np.uint8)
        color_mask = palette[mask]  # shape H,W,3
        return Image.fromarray(color_mask)

    def predict(self, image_path: str):
        """
        Run inference on a single image
        Returns:
            mask: numpy array of class indices
            color_mask_img: PIL Image of colored mask
            overlay_img: PIL Image of original + mask overlay
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]  # shape [num_classes,H,W]

        # Class mask
        mask = output.argmax(0).cpu().numpy().astype(np.uint8)

        # Color mask
        color_mask_img = self.get_color_mask(mask)
        
        # Resize color mask to original image size
        color_mask_img = color_mask_img.resize(image.size, resample=Image.NEAREST)

        # Overlay on original image
        original_img = image.convert("RGBA")
        color_mask_rgba = color_mask_img.convert("RGBA")
        overlay_img = Image.blend(original_img, color_mask_rgba, alpha=0.5)

        return mask, color_mask_img, overlay_img
