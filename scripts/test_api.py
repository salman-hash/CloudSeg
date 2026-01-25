import requests
from pathlib import Path
from PIL import Image

# =========================
# CONFIG
# =========================
serverApi = "http://0.0.0.0:5000/api"
API_URL = f"{serverApi}/segment"  # change to your VM IP if needed

# Image is outside scripts folder, e.g., parent folder of scripts
SCRIPT_DIR = Path(__file__).parent
LOCAL_IMAGE_PATH = SCRIPT_DIR.parent / "input_test.png"  # adjust name if needed

SAVE_DIR = SCRIPT_DIR / "test_output"
SAVE_DIR.mkdir(exist_ok=True)

# =========================
# Verify image exists
# =========================
if not LOCAL_IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {LOCAL_IMAGE_PATH}")

print(f"[INFO] Using local image: {LOCAL_IMAGE_PATH}")

# =========================
# Upload image to API
# =========================
with open(LOCAL_IMAGE_PATH, "rb") as f:
    files = {"file": (LOCAL_IMAGE_PATH.name, f, "image/png")}
    resp = requests.post(API_URL, files=files)

if resp.status_code != 200:
    raise Exception(f"API request failed: {resp.status_code}, {resp.text}")

data = resp.json()
print("[INFO] API Response:")
print(data)

# =========================
# Fetch mask from API
# =========================
mask_url = f"{serverApi}/mask/{data['image_id']}"
mask_resp = requests.get(mask_url)
if mask_resp.status_code == 200:
    mask_local_path = SAVE_DIR / f"{data['image_id']}_mask.png"
    with open(mask_local_path, "wb") as f:
        f.write(mask_resp.content)
    print(f"[INFO] Mask downloaded to {mask_local_path}")

    # Optional: open mask image
    mask_image = Image.open(mask_local_path)
    mask_image.show()
else:
    print(f"[WARNING] Could not fetch mask: {mask_resp.status_code}")
