"""
Pemilihan target + smoothing.

Detektor cuma ngasih daftar kotak per frame. Dia gak punya ingatan: kotak #2
di frame ini belum tentu objek yang sama dengan kotak #2 di frame sebelumnya.
Dua masalah muncul dari situ, dan file ini nanganin dua-duanya:

  1. Kedip-kedip. Kalau tiap frame lo pilih ulang "yang paling deket", dua
     target berjarak mirip bakal saling rebutan dan crosshair-nya getar.
     Solusi: histeresis -- target yang lagi dikunci dikasih diskon jarak.

  2. Gerakan patah-patah. Langsung teleport ke target itu gerakan yang gak
     mungkin dilakuin tangan manusia. Solusi: kontrol proporsional -- tiap
     frame maju sebagian kecil dari sisa jarak, dengan batas kecepatan.
"""
from __future__ import annotations

import math


def pick_target(dets, crosshair, fov_radius, prev=None,
                stickiness=0.35, aim_height=0.5, allowed_cls=(0,)):
    """
    Pilih satu deteksi buat dibidik.

    dets        -- list[Detection] dalam koordinat lokal frame
    crosshair   -- (x, y) posisi crosshair, koordinat lokal frame
    fov_radius  -- abaikan apa pun di luar radius ini (px). Ini yang bikin
                   assist-nya terasa kayak bantuan, bukan auto-snap.
    prev        -- (x, y) titik bidik yang dipilih frame lalu, atau None
    stickiness  -- 0..1, diskon jarak buat target yang deket sama prev

    Balikin (detection, aim_xy) atau (None, None).
    """
    cx, cy = crosshair
    best = None
    best_cost = float("inf")
    best_aim = None

    for d in dets:
        if allowed_cls is not None and d.cls not in allowed_cls:
            continue
        ax, ay = d.aim_point(aim_height)
        dist = math.hypot(ax - cx, ay - cy)
        if dist > fov_radius:
            continue

        cost = dist
        # histeresis: kalau ini kemungkinan besar target yang sama kayak
        # frame lalu, kasih diskon biar dia gak gampang direbut tetangga.
        if prev is not None:
            drift = math.hypot(ax - prev[0], ay - prev[1])
            if drift < max(40.0, d.w * 0.9):
                cost *= (1.0 - stickiness)

        if cost < best_cost:
            best, best_cost, best_aim = d, cost, (ax, ay)

    return best, best_aim


class AimController:
    """
    Kontrol proporsional berbasis WAKTU buat gerakin cursor ke titik bidik.

    Versi naifnya begini:

        step = sisa_jarak * 0.3        # tiap frame

    dan itu bug halus yang hampir semua orang bikin: kalau loop-nya jalan
    30 fps rasanya beda jauh sama pas 120 fps. Perilaku aim-nya jadi
    tergantung seberapa sibuk komputer lo. Perbaikannya pakai smoothing
    eksponensial yang sadar dt:

        alpha = 1 - exp(-dt / tau)

    tau itu "waktu tanggap": kira-kira sekian detik buat nutup ~63% jarak,
    berapa pun frame rate-nya. tau kecil = agresif, tau besar = lembut.
    max_speed dipatok dalam px/DETIK (bukan px/frame) dengan alasan sama.

    Ini PID dengan suku P doang -- ternyata udah cukup buat target yang
    gerakannya halus, dan gak punya masalah osilasi kayak suku D.
    """

    def __init__(self, tau=0.09, max_speed=1400.0, deadzone=3.0):
        self.tau = max(1e-4, tau)
        self.max_speed = max_speed
        self.deadzone = deadzone

    def step(self, cursor, aim, dt):
        """Balikin (dx, dy) yang mesti ditambahkan ke posisi cursor frame ini."""
        dx = aim[0] - cursor[0]
        dy = aim[1] - cursor[1]
        dist = math.hypot(dx, dy)
        if dist <= self.deadzone:
            return 0.0, 0.0

        # dt dijepit: satu frame nyangkut jangan bikin cursor teleport
        dt = min(max(dt, 1e-4), 0.05)
        alpha = 1.0 - math.exp(-dt / self.tau)

        sx, sy = dx * alpha, dy * alpha

        # batasi panjang vektor, bukan tiap sumbu -- kalau per-sumbu,
        # gerakan diagonal jadi belok arahnya.
        max_step = self.max_speed * dt
        mag = math.hypot(sx, sy)
        if mag > max_step:
            k = max_step / mag
            sx, sy = sx * k, sy * k
        return sx, sy


class FpsMeter:
    """Rata-rata bergerak sederhana buat FPS -- angka mentahnya terlalu loncat-loncat."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.fps = 0.0

    def update(self, dt):
        if dt <= 0:
            return self.fps
        inst = 1.0 / dt
        self.fps = inst if self.fps == 0 else (1 - self.alpha) * self.fps + self.alpha * inst
        return self.fps
