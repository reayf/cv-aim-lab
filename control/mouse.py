"""
Kontrol cursor + baca keyboard di Windows, lewat ctypes langsung.

Gak pakai pyautogui/keyboard sengaja: dua-duanya nambah dependency, pyautogui
punya delay bawaan yang bikin loop-nya lemot, dan `keyboard` sering butuh
admin. Win32 API mentah itu lebih cepat dan lebih jelas apa yang terjadi.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

# --- virtual key codes yang kepakai ---------------------------------------
VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
VK_SHIFT = 0x10
VK_CAPITAL = 0x14
VK_ESCAPE = 0x1B
VK_F1, VK_F2, VK_F3, VK_F4 = 0x70, 0x71, 0x72, 0x73

KEY_NAMES = {
    "lmb": VK_LBUTTON, "rmb": VK_RBUTTON, "shift": VK_SHIFT,
    "caps": VK_CAPITAL, "esc": VK_ESCAPE,
    "f1": VK_F1, "f2": VK_F2, "f3": VK_F3, "f4": VK_F4,
}


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _InputUnion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _InputUnion)]


INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001


def get_cursor_pos():
    pt = POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def set_cursor_pos(x, y):
    """
    Pindah cursor ke koordinat layar absolut.

    Cocok buat game yang baca posisi mouse absolut -- termasuk practice range
    kita (pygame.mouse.get_pos). Game FPS 3D biasanya pakai raw input dan
    ngunci cursor di tengah; buat itu yang jalan cuma move_relative().
    """
    user32.SetCursorPos(int(x), int(y))


def move_relative(dx, dy):
    """
    Gerakan mouse relatif lewat SendInput -- kayak lo geser mouse fisik.

    Ini disediain karena inilah bedanya "gerakin cursor" sama "gerakin
    kamera": SetCursorPos itu mindahin penunjuk, SendInput itu nyuntikin
    gerakan perangkat. Buat lab ini SetCursorPos udah cukup.
    """
    dx, dy = int(round(dx)), int(round(dy))
    if dx == 0 and dy == 0:
        return
    inp = INPUT(type=INPUT_MOUSE,
                u=_InputUnion(mi=MOUSEINPUT(dx, dy, 0, MOUSEEVENTF_MOVE, 0, None)))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def is_down(vk: int) -> bool:
    """Tombol lagi ditekan sekarang? (bit tinggi = sedang ditekan)"""
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


class EdgeDetector:
    """
    Deteksi transisi tidak-ditekan -> ditekan.

    Perlu karena loop-nya jalan ratusan kali per detik: tanpa ini, satu
    pencetan tombol kebaca sebagai puluhan toggle.
    """

    def __init__(self, *vks):
        self.state = {vk: False for vk in vks}

    def pressed(self, vk) -> bool:
        now = is_down(vk)
        was = self.state.get(vk, False)
        self.state[vk] = now
        return now and not was


def resolve_key(name: str) -> int:
    key = name.strip().lower()
    if key in KEY_NAMES:
        return KEY_NAMES[key]
    if len(key) == 1 and key.isalnum():
        return ord(key.upper())
    raise ValueError(f"tombol gak dikenal: {name!r} (pilihan: {', '.join(KEY_NAMES)}, atau satu huruf/angka)")
