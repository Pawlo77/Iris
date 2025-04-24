import os
from pathlib import Path
from PIL import Image

SRC_DIR = Path(r"figures")
DST_DIR = SRC_DIR.parent / "figures_small"
MAX_SIDE = 1200  # maksymalna długość boku
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}

DST_DIR.mkdir(exist_ok=True)

for img_path in SRC_DIR.iterdir():
    if img_path.suffix.lower() not in ALLOWED_EXT:
        continue  # pomijamy pliki nie-graficzne

    with Image.open(img_path) as img:
        img = img.convert("RGB")  # unifikacja trybu kolorów
        w, h = img.size
        scale = max(w / MAX_SIDE, h / MAX_SIDE, 1)
        new_size = (int(w / scale), int(h / scale))

        if scale > 1:  # tylko jeśli obraz jest za duży
            img = img.resize(new_size, Image.LANCZOS)

        out_path = DST_DIR / img_path.name
        img.save(out_path, optimize=True, quality=95)
        print(f"{img_path.name}: {w}×{h}  →  {img.size[0]}×{img.size[1]}")

print(f"\nGotowe! Przeskalowane pliki znajdziesz w: {DST_DIR}")
