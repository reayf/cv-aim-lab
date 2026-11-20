"""
Loop utama: capture -> deteksi -> pilih target -> gerakin cursor.

Ini yang nyatuin semuanya. Alurnya per frame:

    1. grab pixel dari region layar
    2. detektor cari kotak (warna atau YOLO)
    3. pilih satu kotak yang paling masuk akal buat dibidik
    4. controller hitung langkah kecil ke arah situ
    5. gerakin cursor -- CUMA kalau tombol assist lagi ditekan
    6. gambar jendela debug biar setiap langkah di atas kelihatan

Secara default dia cuma mau nempel ke jendela practice range kita. Itu
disengaja: lab ini buat belajar CV dengan sasaran dummy buatan sendiri.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from control import mouse
from vision.capture import (
    Region, ScreenCapture, enable_dpi_awareness, region_for_window, wait_for_window,
)
from vision.detector_color import ColorDetector
from vision.tracking import AimController, FpsMeter, pick_target

DEBUG_WIN = "CV Aim Lab -- debug view"


# --------------------------------------------------------------------------
def build_detector(args):
    if args.detector == "yolo":
        from vision.detector_yolo import YoloDetector
        print(f"[detector] YOLO  weights={args.weights}  conf={args.conf}")
        return YoloDetector(args.weights, conf=args.conf, imgsz=args.imgsz,
                            classes=[C.CLS_TARGET] if args.targets_only else None)
    print(f"[detector] warna HSV  h_tol={args.h_tol}  min_area={args.min_area}")
    return ColorDetector(C.TARGET_RGB, h_tol=args.h_tol, s_min=args.s_min,
                         v_min=args.v_min, min_area=args.min_area,
                         min_fill=args.min_fill, cls=C.CLS_TARGET)


def resolve_region(args) -> Region | None:
    if args.region:
        l, t, w, h = args.region
        return Region(l, t, w, h)
    return wait_for_window(args.window, args.wait)


# --------------------------------------------------------------------------
def draw_debug(frame, dets, target, aim, cursor_local, args, fps, active, mask):
    """Gambar overlay debug. Ini bagian paling berharga buat belajar --
    lo bisa lihat langsung kenapa detektornya salah."""
    view = frame.copy()

    # semua deteksi: abu-abu
    for d in dets:
        cv2.rectangle(view, (d.x, d.y), (d.x + d.w, d.y + d.h), (140, 140, 140), 1)

    # lingkaran FOV di sekitar crosshair
    if cursor_local is not None:
        cv2.circle(view, (int(cursor_local[0]), int(cursor_local[1])),
                   int(args.fov), (90, 90, 90), 1)

    # target terpilih: hijau, plus garis ke titik bidik
    if target is not None and aim is not None:
        col = (80, 255, 160) if active else (80, 180, 255)
        cv2.rectangle(view, (target.x, target.y),
                      (target.x + target.w, target.y + target.h), col, 2)
        cv2.circle(view, (int(aim[0]), int(aim[1])), 4, col, -1)
        if cursor_local is not None:
            cv2.line(view, (int(cursor_local[0]), int(cursor_local[1])),
                     (int(aim[0]), int(aim[1])), col, 1, cv2.LINE_AA)
        cv2.putText(view, f"{target.score:.2f}", (target.x, max(14, target.y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    # inset mask (cuma buat detektor warna)
    if mask is not None and args.show_mask:
        h, w = view.shape[:2]
        small = cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        view[h - small.shape[0] - 8:h - 8, w - small.shape[1] - 8:w - 8] = small
        cv2.putText(view, "mask", (w - small.shape[1] - 8, h - small.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # baris status
    state = "ASSIST ON " if active else "idle      "
    scol = (80, 255, 160) if active else (150, 150, 150)
    bar = f"{state} | {fps:5.1f} fps | deteksi: {len(dets):2d} | tahan {args.key.upper()}"
    cv2.rectangle(view, (0, 0), (view.shape[1], 30), (18, 18, 22), -1)
    cv2.putText(view, bar, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, scol, 1, cv2.LINE_AA)
    cv2.putText(view, "F2 hold/toggle   F3 mask   ESC keluar",
                (view.shape[1] - 340, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (140, 140, 150), 1, cv2.LINE_AA)
    return view


# --------------------------------------------------------------------------
def run(args):
    enable_dpi_awareness()

    region = resolve_region(args)
    if region is None:
        print(f"[!] Jendela '{args.window}' gak ketemu.\n"
              f"    Jalanin dulu di terminal lain:  python main.py range\n"
              f"    Atau tunjuk region manual:      --region L T W H")
        return 1
    print(f"[region] {region.width}x{region.height} @ ({region.left},{region.top})")

    detector = build_detector(args)
    controller = AimController(args.tau, args.max_speed, args.deadzone)
    fps_meter = FpsMeter()
    edges = mouse.EdgeDetector(mouse.VK_F2, mouse.VK_F3, mouse.VK_ESCAPE)
    assist_key = mouse.resolve_key(args.key)

    cap = ScreenCapture()
    cv2.namedWindow(DEBUG_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEBUG_WIN, region.width // 2, region.height // 2)

    prev_aim = None
    always_on = False
    last_t = time.perf_counter()
    frame_i = 0

    print("\n  tahan " + args.key.upper() + " = assist aktif   |   F2 = kunci nyala terus"
          "   |   F3 = mask   |   ESC = keluar\n")

    try:
        while True:
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            fps = fps_meter.update(dt)
            frame_i += 1

            # jendela bisa dipindah -> refresh region-nya sesekali
            if not args.region and frame_i % 60 == 0:
                fresh = region_for_window(args.window)
                if fresh is not None:
                    region = fresh

            if edges.pressed(mouse.VK_ESCAPE):
                break
            if edges.pressed(mouse.VK_F2):
                always_on = not always_on
                print(f"[assist] {'NYALA TERUS' if always_on else 'mode tahan-tombol'}")
            if edges.pressed(mouse.VK_F3):
                args.show_mask = not args.show_mask

            frame = cap.grab(region)
            dets, mask = detector.detect(frame)

            cursor = mouse.get_cursor_pos()
            cursor_local = region.to_local(*cursor)
            in_region = region.contains(*cursor)

            target, aim = pick_target(
                dets, cursor_local, args.fov, prev_aim,
                stickiness=args.stickiness, aim_height=args.aim_height,
                allowed_cls=(C.CLS_TARGET,) if args.targets_only else None,
            )
            prev_aim = aim

            active = (always_on or mouse.is_down(assist_key)) and in_region
            if active and aim is not None:
                dx, dy = controller.step(cursor_local, aim, dt)
                if dx or dy:
                    sx, sy = region.to_screen(cursor_local[0] + dx, cursor_local[1] + dy)
                    mouse.set_cursor_pos(sx, sy)

            view = draw_debug(frame, dets, target, aim, cursor_local if in_region else None,
                              args, fps, active, mask)
            cv2.imshow(DEBUG_WIN, view)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            if cv2.getWindowProperty(DEBUG_WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.close()
        cv2.destroyAllWindows()
    return 0


# --------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        description="CV Aim Lab -- loop aim assist berbasis computer vision",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = ap.add_argument_group("sumber gambar")
    src.add_argument("--window", default=C.WINDOW_TITLE,
                     help="potongan judul jendela yang mau di-capture")
    src.add_argument("--region", type=int, nargs=4, metavar=("L", "T", "W", "H"),
                     help="capture region layar manual, bukan cari jendela")
    src.add_argument("--wait", type=float, default=15.0,
                     help="detik menunggu jendela game muncul")

    det = ap.add_argument_group("detektor")
    det.add_argument("--detector", choices=["color", "yolo"], default="color")
    det.add_argument("--weights", default="runs/detect/train/weights/best.pt")
    det.add_argument("--conf", type=float, default=0.35, help="ambang confidence YOLO")
    det.add_argument("--imgsz", type=int, default=640)
    det.add_argument("--h-tol", dest="h_tol", type=int, default=12,
                     help="setengah lebar band hue (0-90)")
    det.add_argument("--s-min", dest="s_min", type=int, default=90)
    det.add_argument("--v-min", dest="v_min", type=int, default=70)
    det.add_argument("--min-area", dest="min_area", type=int, default=80)
    det.add_argument("--min-fill", dest="min_fill", type=float, default=0.45)
    det.add_argument("--all-classes", dest="targets_only", action="store_false",
                     help="ikut bidik decoy juga (buat lihat efek salah kelas)")

    aim = ap.add_argument_group("perilaku aim")
    aim.add_argument("--fov", type=float, default=C.FOV_RADIUS)
    aim.add_argument("--tau", type=float, default=C.AIM_TAU,
                     help="waktu tanggap dalam detik (kecil = agresif)")
    aim.add_argument("--max-speed", dest="max_speed", type=float, default=C.MAX_SPEED,
                     help="batas kecepatan cursor, px/detik")
    aim.add_argument("--deadzone", type=float, default=C.DEADZONE)
    aim.add_argument("--aim-height", dest="aim_height", type=float, default=C.AIM_HEIGHT)
    aim.add_argument("--stickiness", type=float, default=C.STICKINESS)
    aim.add_argument("--key", default="rmb", help="tombol tahan buat aktifin assist")
    aim.add_argument("--show-mask", dest="show_mask", action="store_true", default=True)
    return ap


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
