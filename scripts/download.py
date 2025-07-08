import os
import requests
from dotenv import load_dotenv

# 1. Load .env
load_dotenv()  
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
if not UNSPLASH_ACCESS_KEY:
    raise ValueError("Missing UNSPLASH_ACCESS_KEY in environment")

# 2. Thông số chung
QUERY     = "dreamy landscape"
SAVE_DIR  = "data/raw"
PER_PAGE  = 30  # ảnh/trang
TOTAL_PG   = 6  # số trang muốn tải

os.makedirs(SAVE_DIR, exist_ok=True)

def download_image(url, path):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)

def fetch_from_unsplash(page):
    api = "https://api.unsplash.com/search/photos"
    params = {
        "query": QUERY,
        "page": page,
        "per_page": PER_PAGE,
        "client_id": UNSPLASH_ACCESS_KEY
    }
    data = requests.get(api, params=params).json()
    for i, photo in enumerate(data.get("results", [])):
        url       = photo["urls"]["regular"]
        filename  = f"img_p{page}_{i}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)
        download_image(url, save_path)
        print(f"[Page {page}] downloaded {filename}")

if __name__ == "__main__":
    for pg in range(1, TOTAL_PG + 1):
        fetch_from_unsplash(pg)
