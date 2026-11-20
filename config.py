"""
Konfigurasi bersama untuk CV Aim Lab.

Game latihan dan pipeline computer vision dua-duanya baca file ini, jadi
"seperti apa sih rupanya target itu" cuma didefinisikan di SATU tempat.
Detector menurunkan rentang HSV-nya dari TARGET_RGB di bawah, bukan dari
angka ajaib yang di-hardcode.
"""

# --- Jendela practice range ------------------------------------------------
WINDOW_TITLE = "CV Aim Lab - Practice Range"
GAME_W, GAME_H = 1280, 720
GAME_FPS = 120

# Area main (inset dari tepi jendela). Sisanya buat HUD.
PLAY_MARGIN_X = 40
PLAY_TOP = 70
PLAY_BOTTOM = 40

# --- Warna (RGB) -----------------------------------------------------------
# Target magenta: hue-nya jauh dari semua warna scenery, tapi TIDAK terlalu
# jauh dari decoy oranye -- itu sengaja, biar lo harus nyetel hue band-nya.
TARGET_RGB = (228, 32, 196)
DECOY_RGB = (232, 122, 40)
TARGET_RING_RGB = (252, 226, 250)
DECOY_RING_RGB = (255, 236, 200)

BG_RGB = (24, 26, 32)
GRID_RGB = (38, 42, 52)
PILLAR_RGB = (58, 62, 74)
PILLAR_EDGE_RGB = (86, 92, 108)
HUD_RGB = (206, 212, 224)
HUD_DIM_RGB = (118, 126, 142)
CROSSHAIR_RGB = (120, 255, 190)
HIT_RGB = (120, 255, 190)
MISS_RGB = (255, 110, 110)

# --- Aturan spawn ----------------------------------------------------------
N_TARGETS = 5
N_DECOYS = 4
RADIUS_MIN, RADIUS_MAX = 18, 42
SPEED_MIN, SPEED_MAX = 40.0, 190.0   # pixel per detik
N_PILLARS = 3

# --- Kelas dataset ---------------------------------------------------------
CLASS_NAMES = ["target", "decoy"]
CLS_TARGET, CLS_DECOY = 0, 1

# --- Default aim assist ----------------------------------------------------
FOV_RADIUS = 220        # cuma bidik yang jaraknya <= sekian px dari crosshair
AIM_TAU = 0.09          # detik -- waktu tanggap, gak tergantung frame rate
MAX_SPEED = 1400.0      # px/detik -- batas kecepatan cursor
DEADZONE = 3            # udah cukup deket -> diem
AIM_HEIGHT = 0.5        # 0 = atas bbox, 0.5 = tengah, 1 = bawah
STICKINESS = 0.35       # diskon jarak buat target yang lagi dikunci
