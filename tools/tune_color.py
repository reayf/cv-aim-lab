"""
Penyetel band HSV interaktif.

Ini alat belajar yang paling cepat ngasih "aha". Dia capture jendela range
secara live dan kasih slider buat tiap parameter detektor. Geser h_tol pelan-
pelan sampai decoy oranye mulai ikut kedetek, dan lo bakal paham kenapa
threshold warna itu rapuh -- yang pada akhirnya bikin lo ngerti kenapa orang
repot-repot pakai YOLO.

Angka yang muncul di terminal pas keluar tinggal lo tempel jadi flag
`python main.py assist`.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from vision.capture import (
    Region, ScreenCapture, enable_dpi_awareness, region_for_window, wait_for_window,
)
from vision.detector_color import ColorDetector

WIN = "tune -- band warna HSV"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Setel band warna detektor secara live")
    ap.add_argument("--window", default=C.WINDOW_TITLE)
    ap.add_argument("--region", type=int, nargs=4, metavar=("L", "T", "W", "H"))
    ap.add_argument("--wait", type=float, default=15.0,
                    help="detik menunggu jendelanya muncul")
    a = ap.parse_args(argv)

    enable_dpi_awareness()
    region = Region(*a.region) if a.region else wait_for_window(a.window, a.wait)
    if region is None:
        print(f"[!] Jendela '{a.window}' gak ketemu. Jalanin dulu: python main.py range")
        return 1

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, region.width // 2, region.height // 2 + 60)
    for name, val, maxi in [("h_tol", 12, 90), ("s_min", 90, 255), ("v_min", 70, 255),
                            ("min_area", 80, 2000), ("min_fill x100", 45, 100)]:
        cv2.createTrackbar(name, WIN, val, maxi, lambda _v: None)

    print("\n  Geser slidernya. Kotak hijau = kedetek.")
    print("  Naikin h_tol sampai bola oranye ikut ke-highlight -- di situ band-nya kelebaran.")
    print("  ESC buat keluar dan cetak flag-nya.\n")

    cap = ScreenCapture()
    last = {}
    frame_i = 0
    lost = False
    try:
        while True:
            # Jendela game bisa digeser, di-resize, atau ditutup sambil lo
            # nyetel. Tanpa refresh, tuner-nya diem-diem ngecapture area yang
            # salah dan lo bakal bingung kenapa mask-nya kosong melompong.
            frame_i += 1
            if not a.region and frame_i % 30 == 0:
                fresh = region_for_window(a.window)
                if fresh is not None:
                    if lost:
                        print("[region] jendela game ketemu lagi.")
                        lost = False
                    region = fresh
                elif not lost:
                    print("[region] jendela game ilang -- nyalain lagi: python main.py range")
                    lost = True

            h_tol = max(1, cv2.getTrackbarPos("h_tol", WIN))
            s_min = cv2.getTrackbarPos("s_min", WIN)
            v_min = cv2.getTrackbarPos("v_min", WIN)
            min_area = max(1, cv2.getTrackbarPos("min_area", WIN))
            min_fill = cv2.getTrackbarPos("min_fill x100", WIN) / 100.0
            last = dict(h_tol=h_tol, s_min=s_min, v_min=v_min,
                        min_area=min_area, min_fill=min_fill)

            det = ColorDetector(C.TARGET_RGB, h_tol=h_tol, s_min=s_min, v_min=v_min,
                                min_area=min_area, min_fill=min_fill)
            frame = cap.grab(region)
            dets, mask = det.detect(frame)

            view = frame.copy()
            for d in dets:
                cv2.rectangle(view, (d.x, d.y), (d.x + d.w, d.y + d.h), (80, 255, 160), 2)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            h = 360
            scale = h / view.shape[0]
            side = cv2.hconcat([
                cv2.resize(view, (int(view.shape[1] * scale), h)),
                cv2.resize(mask_bgr, (int(mask_bgr.shape[1] * scale), h)),
            ])
            cv2.putText(side, f"{len(dets)} deteksi   (kiri: frame + box | kanan: mask)",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 160), 1, cv2.LINE_AA)
            cv2.imshow(WIN, side)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.close()
        cv2.destroyAllWindows()

    if last:
        print("\nSetelan lo, tinggal tempel:\n")
        print(f"  python main.py assist --h-tol {last['h_tol']} --s-min {last['s_min']} "
              f"--v-min {last['v_min']} --min-area {last['min_area']} "
              f"--min-fill {last['min_fill']:.2f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
