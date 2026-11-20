"""
Detector baseline: segmentasi warna HSV + analisis kontur.

Ini pipeline CV klasik, dan sengaja jadi yang pertama lo pelajari sebelum
lompat ke YOLO. Urutannya:

    BGR -> HSV -> threshold -> morfologi -> kontur -> filter bentuk -> bbox

Kenapa HSV bukan RGB? Di RGB, "magenta terang" dan "magenta gelap" itu tiga
angka yang beda semua. Di HSV, warnanya (H) tetap sama, cuma terang (V) dan
pekat (S) yang berubah. Jadi lo bisa bilang "ambil semua yang hue-nya sekitar
150" dan itu tahan terhadap perubahan pencahayaan.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    """Satu objek terdeteksi, koordinat lokal frame."""
    x: int            # kiri bbox
    y: int            # atas bbox
    w: int
    h: int
    score: float      # 0..1, keyakinan
    cls: int = 0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> int:
        return self.w * self.h

    def aim_point(self, height_ratio: float = 0.5):
        """
        Titik bidik di dalam bbox. 0.0 = pucuk atas, 0.5 = tengah, 1.0 = dasar.
        Di FPS sungguhan orang pakai ~0.25 buat "headshot"; di sini target
        bulat, jadi 0.5 yang paling masuk akal.
        """
        return (self.cx, self.y + self.h * height_ratio)


def hsv_bands_from_rgb(rgb, h_tol=12, s_min=90, v_min=70):
    """
    Turunkan rentang HSV dari satu warna RGB acuan.

    Ini kenapa config.TARGET_RGB itu satu-satunya sumber kebenaran: game
    gambar pakai warna itu, detector nurunin threshold-nya dari warna yang
    sama. Ganti warnanya di config, dua-duanya ikut, gak ada yang perlu
    di-tune ulang manual.

    Hue di OpenCV itu 0..179 dan MELINGKAR (merah ada di 0 dan di 179).
    Kalau rentangnya nabrak batas, kita balikin dua band terpisah.
    """
    bgr = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    h, _s, _v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0].tolist()

    lo_h, hi_h = h - h_tol, h + h_tol
    bands = []
    if lo_h < 0:                       # melingkar lewat 0
        bands.append((np.array([0, s_min, v_min], np.uint8),
                      np.array([hi_h, 255, 255], np.uint8)))
        bands.append((np.array([180 + lo_h, s_min, v_min], np.uint8),
                      np.array([179, 255, 255], np.uint8)))
    elif hi_h > 179:                   # melingkar lewat 179
        bands.append((np.array([lo_h, s_min, v_min], np.uint8),
                      np.array([179, 255, 255], np.uint8)))
        bands.append((np.array([0, s_min, v_min], np.uint8),
                      np.array([hi_h - 180, 255, 255], np.uint8)))
    else:
        bands.append((np.array([lo_h, s_min, v_min], np.uint8),
                      np.array([hi_h, 255, 255], np.uint8)))
    return bands


class ColorDetector:
    """
    Detektor blob berbasis warna.

    Knob yang paling ngaruh, urut dari yang paling sering perlu disetel:
      h_tol     -- lebar band hue. Kegedean -> decoy oranye ikut kedetek.
                   Kekecilan -> tepi target yang ke-antialias hilang.
      min_area  -- buang bintik noise.
      min_fill  -- luas kontur / luas bbox. Lingkaran penuh ~0.78, garis
                   tipis / coretan HUD jauh lebih kecil. Filter bentuk yang
                   murah tapi ampuh banget.
    """

    def __init__(self, rgb, h_tol=12, s_min=90, v_min=70,
                 min_area=80, max_area=200_000, min_fill=0.45, cls=0):
        self.bands = hsv_bands_from_rgb(rgb, h_tol, s_min, v_min)
        self.min_area = min_area
        self.max_area = max_area
        self.min_fill = min_fill
        self.cls = cls
        # kernel elips 5x5: buang bintik nyasar, tambal lubang kecil
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        m = None
        for lo, hi in self.bands:
            band = cv2.inRange(hsv, lo, hi)
            m = band if m is None else cv2.bitwise_or(m, band)
        # OPEN buang noise garam-merica, CLOSE nambal lubang di dalam blob
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        return m

    def detect(self, frame_bgr: np.ndarray):
        """Balikin (list[Detection], mask) -- mask-nya buat di-debug secara visual."""
        m = self.mask(frame_bgr)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dets = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            fill = area / max(1.0, float(w * h))
            if fill < self.min_fill:
                continue
            # target kita bulat -> bbox-nya harusnya mendekati bujur sangkar.
            aspect = w / max(1.0, float(h))
            if not (0.45 < aspect < 2.2):
                continue
            # "score" = seberapa mirip lingkaran. Lingkaran sempurna = pi/4 = 0.785
            score = min(1.0, fill / 0.785)
            dets.append(Detection(x, y, w, h, score, self.cls))

        dets.sort(key=lambda d: d.area, reverse=True)
        return dets, m


def draw_detections(frame, dets, color=(0, 255, 120), thickness=2, label=None):
    """Gambar kotak deteksi ke frame (buat jendela debug)."""
    for d in dets:
        cv2.rectangle(frame, (d.x, d.y), (d.x + d.w, d.y + d.h), color, thickness)
        if label:
            cv2.putText(frame, f"{label} {d.score:.2f}", (d.x, max(12, d.y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return frame
