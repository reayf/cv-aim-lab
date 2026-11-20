"""
Detector YOLO -- langkah kedua setelah lo paham detector warna.

Bedanya sama detector_color.py:
  - Detector warna: LO yang nulis aturannya ("hue sekitar 150, bulat, > 80px").
    Cepat, transparan, tapi patah begitu warnanya berubah atau bentuknya beda.
  - YOLO: aturannya DIPELAJARI dari contoh berlabel. Lebih tahan banting, bisa
    bedain kelas yang mirip, tapi butuh dataset + training + lebih berat.

Pakai model yang dilatih dari `python main.py train`, yang datanya diambil
dari `python main.py range --dataset`.
"""
from __future__ import annotations

import os

import numpy as np

from .detector_color import Detection


class YoloDetector:
    def __init__(self, weights: str, conf=0.35, iou=0.45, imgsz=640,
                 classes=None, device=None):
        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"Model gak ketemu: {weights}\n"
                "Bikin dulu datasetnya lalu latih:\n"
                "  python main.py range --dataset --frames 800\n"
                "  python main.py train"
            )
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("butuh ultralytics: python -m pip install ultralytics") from e

        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes      # None = semua kelas, atau list id kelas
        self.device = device

    def detect(self, frame_bgr: np.ndarray):
        """Balikin (list[Detection], None) -- signature-nya sama kayak ColorDetector."""
        res = self.model.predict(
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )[0]

        dets = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                dets.append(Detection(int(x1), int(y1),
                                      int(x2 - x1), int(y2 - y1),
                                      float(c), int(k)))

        dets.sort(key=lambda d: d.score, reverse=True)
        return dets, None
