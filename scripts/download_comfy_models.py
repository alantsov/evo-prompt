#!/usr/bin/env python3
"""Downloads required Krea-2 models for the basic ComfyUI workflow."""
import sys
import requests
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "ComfyUI" / "models"
HF_BASE = "https://huggingface.co/Comfy-Org/Krea-2/resolve/main"

MODELS = [
    ("diffusion_models", "krea2_turbo_bf16.safetensors"),
    ("text_encoders", "qwen3vl_4b_bf16.safetensors"),
    ("vae", "qwen_image_vae.safetensors"),
]

def download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"⏭️  {dest.name} already exists. Skipping.")
        return True

    print(f"📥 Downloading {dest.name}...")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0

            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = (downloaded / total) * 100
                        sys.stdout.write(f"\r   [{pct:5.1f}%] {downloaded/1e6:.1f}/{total/1e6:.1f} MB")
                        sys.stdout.flush()

            sys.stdout.write(f"\r   [100.0%] {total/1e6:.1f}/{total/1e6:.1f} MB ✅\n")
            return True
    except Exception as e:
        print(f"\n❌ Failed to download {dest.name}: {e}")
        if dest.exists():
            dest.unlink()
        return False

def main() -> None:
    print("📦 Setting up Krea-2 basic workflow models...\n")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    for folder, filename in MODELS:
        target_dir = MODELS_DIR / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        if download_file(f"{HF_BASE}/{folder}/{filename}", target_dir / filename):
            success += 1

    print(f"\n{'='*40}")
    print(f"✅ Done. {success}/{len(MODELS)} models ready in {MODELS_DIR}")
    if success < len(MODELS):
        print("⚠️  Some downloads failed. Check your connection or re-run.")

if __name__ == "__main__":
    main()
