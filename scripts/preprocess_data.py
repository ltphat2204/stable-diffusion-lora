import os
from PIL import Image
from tqdm import tqdm

# Thư mục input/output
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
TARGET_SIZE   = (512, 512)  # Kích thước chuẩn

os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess_image(in_path, out_path):
    img = Image.open(in_path).convert("RGB")
    img = img.resize(TARGET_SIZE, resample=Image.LANCZOS)

    img.save(out_path, format="JPEG", quality=95)

def main():
    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith((".jpg",".png"))]
    for fname in tqdm(files, desc="Preprocessing images"):
        src = os.path.join(RAW_DIR, fname)
        dst = os.path.join(PROCESSED_DIR, fname)
        preprocess_image(src, dst)
    print(f"Done! {len(files)} images saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
