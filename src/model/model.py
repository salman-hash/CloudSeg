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

    def predict(self, image_path):
        """
        Run inference on a single image
        Returns: segmentation mask as numpy array
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]  # shape [21,H,W]

        # Convert to class mask
        mask = output.argmax(0).cpu().numpy().astype(np.uint8)
        return mask
