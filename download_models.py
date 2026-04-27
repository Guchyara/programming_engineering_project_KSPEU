import requests
import os

def download_model(url: str, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Downloading model from {url} ...")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with open(save_path, "wb") as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"\rDownloaded {downloaded / total * 100:.2f}%", end="")

    print(f"\nSaved to {save_path}")
