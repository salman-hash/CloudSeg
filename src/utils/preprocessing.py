from torchvision import transforms
from PIL import Image

def get_preprocessing(input_size=(512, 512)):
    """
    Returns a composed preprocessing transform for DeepLabV3 model.
    """
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def resize_mask(mask, size=(512, 512)):
    """
    Resize a mask (numpy or PIL) to given size.
    """
    if isinstance(mask, Image.Image):
        return mask.resize(size)
    else:
        from PIL import Image
        mask_img = Image.fromarray(mask.astype('uint8'))
        return mask_img.resize(size)