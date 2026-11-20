"""
Latih YOLO dari dataset auto-label yang direkam practice range.

Yang dikerjain:
  1. Bagi dataset/ jadi train/val (default 85/15)
  2. Tulis data.yaml yang dibutuhin ultralytics
  3. Jalanin training

Split-nya pakai symlink kalau bisa, kalau gak bisa (butuh admin di Windows)
otomatis fallback ke copy file.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

IMG_EXT = ".png"


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except (OSError, NotImplementedError, AttributeError):
        shutil.copy2(src, dst)


def prepare_split(dataset_dir, out_dir, val_frac=0.15, seed=0):
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Dataset gak ketemu di {dataset_dir}.\n"
            "Rekam dulu:  python main.py range --dataset --frames 800"
        )

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(images_dir)
                   if f.endswith(IMG_EXT))
    if not stems:
        raise RuntimeError(f"gak ada gambar {IMG_EXT} di {images_dir}")

    random.Random(seed).shuffle(stems)
    n_val = max(1, int(len(stems) * val_frac))
    splits = {"val": stems[:n_val], "train": stems[n_val:]}

    for split, names in splits.items():
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(out_dir, split, sub), exist_ok=True)
        for stem in names:
            link_or_copy(os.path.join(images_dir, stem + IMG_EXT),
                         os.path.join(out_dir, split, "images", stem + IMG_EXT))
            lbl = os.path.join(labels_dir, stem + ".txt")
            if os.path.exists(lbl):
                link_or_copy(lbl, os.path.join(out_dir, split, "labels", stem + ".txt"))

    yaml_path = os.path.abspath(os.path.join(out_dir, "data.yaml"))
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(C.CLASS_NAMES))
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {os.path.abspath(out_dir)}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"names:\n{names_block}\n"
        )

    print(f"[split] train={len(splits['train'])}  val={len(splits['val'])}")
    print(f"[split] data.yaml -> {yaml_path}")
    return yaml_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Latih YOLO buat target practice range")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="dataset_split")
    ap.add_argument("--model", default="yolo11n.pt", help="checkpoint awal")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="'cpu', '0', dst. (auto kalau kosong)")
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.15)
    ap.add_argument("--prepare-only", action="store_true", help="cuma bikin split, gak training")
    a = ap.parse_args(argv)

    yaml_path = prepare_split(a.dataset, a.out, a.val_frac)
    if a.prepare_only:
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("butuh ultralytics:  python -m pip install ultralytics")
        return 1

    print(f"[train] {a.model}  epochs={a.epochs}  imgsz={a.imgsz}  batch={a.batch}")
    print("[train] di CPU ini bakal lama. Coba --epochs 15 --imgsz 416 dulu buat tes cepat.")
    model = YOLO(a.model)
    model.train(data=yaml_path, epochs=a.epochs, imgsz=a.imgsz,
                batch=a.batch, device=a.device, plots=True)

    print("\nSelesai. Pakai bobotnya:")
    print("  python main.py assist --detector yolo --weights runs/detect/train/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
