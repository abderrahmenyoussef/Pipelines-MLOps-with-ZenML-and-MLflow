import os
import shutil
from pathlib import Path

def main():
    base = Path("data")
    coco = base / "coco128"
    tiny = base / "tiny_coco"

    images_src = coco / "images"
    labels_src = coco / "labels"

    if not images_src.exists() or not labels_src.exists():
        print("[make_tiny_person_from_coco128] Le dossier data/coco128 est manquant.")
        print("Placez le dataset COCO128 dans data/coco128 (images/ et labels/).")
        return

    # On recrée un petit dataset
    if tiny.exists():
        shutil.rmtree(tiny)
    (tiny / "images" / "train").mkdir(parents=True, exist_ok=True)
    (tiny / "images" / "val").mkdir(parents=True, exist_ok=True)
    (tiny / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tiny / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Copier les 50 premières images/labels pour l'exemple
    img_files = sorted(images_src.glob("*.jpg"))[:50]
    for i, img in enumerate(img_files):
        lbl = labels_src / (img.stem + ".txt")
        if not lbl.exists():
            continue

        split = "train" if i < 40 else "val"
        shutil.copy2(img, tiny / "images" / split / img.name)
        shutil.copy2(lbl, tiny / "labels" / split / lbl.name)

    print("[make_tiny_person_from_coco128] Dataset tiny_coco généré dans data/tiny_coco")

if __name__ == "__main__":
    main()
