"""
Screen capture + pencarian jendela (Windows).

Ini "mata" dari pipeline. Dia gak tau apa-apa soal isi game -- cuma ngambil
pixel dari sebuah persegi di layar, persis kayak mata manusia. Semua yang
tau soal game cuma ada di file ini sebatas: judul jendelanya apa.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

import cv2
import numpy as np
from mss import mss

user32 = ctypes.WinDLL("user32", use_last_error=True)


def enable_dpi_awareness():
    """
    Tanpa ini, di layar ber-scaling (125%, 150%) koordinat jendela yang
    dilaporkan Windows bakal beda sama pixel asli -> box deteksi meleset.
    Panggil ini SEBELUM bikin jendela apa pun.
    """
    for fn in (
        lambda: ctypes.windll.user32.SetProcessDpiAwarenessContext(-4),  # per-monitor v2
        lambda: ctypes.windll.shcore.SetProcessDpiAwareness(2),
        lambda: ctypes.windll.user32.SetProcessDPIAware(),
    ):
        try:
            fn()
            return
        except Exception:
            continue


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


@dataclass(frozen=True)
class Region:
    """Persegi di koordinat layar."""
    left: int
    top: int
    width: int
    height: int

    @property
    def center(self):
        return (self.left + self.width // 2, self.top + self.height // 2)

    def to_screen(self, x, y):
        """Koordinat lokal frame -> koordinat layar absolut."""
        return (self.left + x, self.top + y)

    def to_local(self, x, y):
        """Koordinat layar absolut -> koordinat lokal frame."""
        return (x - self.left, y - self.top)

    def contains(self, x, y):
        return (self.left <= x < self.left + self.width
                and self.top <= y < self.top + self.height)

    def as_dict(self):
        return {"left": self.left, "top": self.top,
                "width": self.width, "height": self.height}


def find_window(title_substring: str):
    """Cari HWND pertama yang judulnya mengandung title_substring (case-insensitive)."""
    found = []
    needle = title_substring.lower()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n:
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if needle in buf.value.lower():
                found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(cb, 0)
    return found[0] if found else (None, None)


def client_region(hwnd) -> Region | None:
    """
    Area klien jendela (tanpa title bar / border) dalam koordinat layar.
    Dipakai supaya frame yang kita capture 1:1 sama surface game.
    """
    if not hwnd:
        return None
    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    origin = POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None
    return Region(origin.x, origin.y, w, h)


def region_for_window(title_substring: str) -> Region | None:
    hwnd, _ = find_window(title_substring)
    return client_region(hwnd)


def wait_for_window(title_substring: str, timeout: float = 15.0,
                    poll: float = 0.25, verbose: bool = True) -> Region | None:
    """
    Tunggu jendelanya muncul, maksimal `timeout` detik.

    Alurnya dua terminal, jadi urutan start-nya gak bisa dijamin. Tanpa ini
    lo harus selalu inget nyalain game duluan, dan kalau kecepetan sedetik
    doang programnya langsung nyerah. Nunggu bentar jauh lebih enak dipakai.
    """
    import time as _time

    region = region_for_window(title_substring)
    if region is not None:
        return region

    if verbose:
        print(f"[tunggu] nyari jendela '{title_substring}' "
              f"(maks {timeout:.0f}s) -- nyalain kalau belum...")
    deadline = _time.perf_counter() + timeout
    while _time.perf_counter() < deadline:
        _time.sleep(poll)
        region = region_for_window(title_substring)
        if region is not None:
            if verbose:
                print("[tunggu] ketemu.")
            return region
    return None


def primary_monitor_region() -> Region:
    with mss() as sct:
        m = sct.monitors[1]
    return Region(m["left"], m["top"], m["width"], m["height"])


class ScreenCapture:
    """
    Pembungkus tipis di atas mss.

    mss harus dipakai di thread yang sama dengan tempat dia dibuat, makanya
    objeknya disimpan, bukan bikin baru tiap frame (bikin baru tiap frame
    itu bottleneck utama yang bikin capture loop pemula lemot).
    """

    def __init__(self):
        self._sct = mss()

    def grab(self, region: Region) -> np.ndarray:
        """
        Ambil frame, balikin array BGR (H, W, 3) uint8 -- format standar OpenCV.

        Catatan performa yang mahal dipelajarinya: JANGAN pakai
        `np.asarray(shot)` di objek ScreenShot-nya mss. Dia gak punya buffer
        protocol, jadi numpy jatuh ke iterasi elemen-per-elemen dan itu makan
        ~20 ms buat frame 720p -- lebih lama dari capture-nya sendiri.
        `np.frombuffer` di atas .raw itu view zero-copy, dan cvtColor buang
        channel alpha-nya dengan kode SIMD. Total jadi < 2 ms.
        """
        shot = self._sct.grab(region.as_dict())
        bgra = np.frombuffer(shot.raw, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    def close(self):
        try:
            self._sct.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
