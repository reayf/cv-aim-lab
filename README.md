# CV Aim Lab

Lab belajar computer vision pakai target dummy. Isinya dua bagian yang sengaja
dipisah total:

1. **Practice Range** — game tembak-tembakan offline buatan sendiri. Ini yang
   jadi "dunia" buat dilihat.
2. **Pipeline CV** — program terpisah yang cuma lihat *pixel dari layar*, cari
   target, dan gerakin cursor ke situ.

Yang penting: pipeline CV-nya **gak tau apa-apa soal isi game**. Dia gak baca
memory, gak baca variabel, gak nyentuh proses game. Dia cuma lihat gambar,
persis kayak mata lo. Itu yang bikin ini latihan CV beneran — dan itu juga
kenapa lab ini pakai sasaran dummy buatan sendiri, bukan game orang.

---

## Install

```bash
python -m pip install -r requirements.txt
```

Buat jalur YOLO (opsional, nanti aja):

```bash
python -m pip install -r requirements-yolo.txt
```

> Catatan Python 3.14: pakai `pygame-ce`, bukan `pygame`. Isinya sama persis
> (`import pygame`), tapi cuma dia yang punya wheel buat 3.14.

---

## Jalanin

Buka **dua terminal**.

```bash
# terminal 1 -- game-nya
python main.py range

# terminal 2 -- mata + tangannya
python main.py assist
```

Di jendela game: **tahan klik kanan**. Cursor bakal ditarik pelan ke target
magenta terdekat. Lepas, kontrol balik ke lo sepenuhnya.

| Tombol | Fungsi |
|---|---|
| Tahan RMB | assist aktif |
| F2 | kunci nyala terus (gak usah tahan) |
| F3 | tampilkan/sembunyikan inset mask |
| ESC | keluar |

Jendela **debug view** itu bagian paling berharga. Di situ kelihatan:
kotak abu-abu = semua yang kedetek, kotak hijau = target terpilih, lingkaran =
FOV, garis = arah tarikan, pojok kanan bawah = mask hitam-putih mentahnya.
Kalau aim-nya salah, jawabannya selalu kelihatan di jendela ini.

---

## Alur pipeline-nya

```
       ┌─ Practice Range (proses terpisah) ─┐
       │  gambar target ke layar            │
       └────────────────┬───────────────────┘
                        │  pixel di monitor
                        ▼
   grab region  ──►  detektor  ──►  pilih target  ──►  controller  ──►  cursor
   (mss)             (HSV/YOLO)     (FOV+histeresis)   (eksponensial)
   capture.py        detector_*.py  tracking.py        tracking.py      mouse.py
```

Tiap kotak itu satu file, dan tiap file bisa dites sendiri-sendiri.

---

## Peta file

| File | Isinya |
|---|---|
| `config.py` | Satu-satunya sumber kebenaran soal warna & ukuran. Game **dan** detektor dua-duanya baca ini. |
| `range_game/target_range.py` | Game-nya + mode auto-label dataset. |
| `vision/capture.py` | Cari jendela (Win32), DPI awareness, grab pixel. |
| `vision/detector_color.py` | Detektor baseline: HSV → threshold → morfologi → kontur. |
| `vision/detector_yolo.py` | Detektor YOLO, interface-nya sama persis. |
| `vision/tracking.py` | Pilih target (FOV + histeresis) + controller aim. |
| `control/mouse.py` | Cursor & keyboard lewat ctypes Win32 langsung. |
| `control/assist.py` | Loop utama + jendela debug. |
| `tools/tune_color.py` | Penyetel HSV interaktif pakai slider. |
| `tools/train_yolo.py` | Split dataset + training. |

---

## Jalur belajarnya

### Tahap 1 — Ngerti kenapa HSV, bukan RGB

```bash
python main.py tune
```

Slider live buat tiap parameter detektor. Yang wajib dicoba: **naikin `h_tol`
pelan-pelan**. Bola oranye bakal tetep aman... sampai tiba-tiba **di `h_tol`
38** semuanya ikut ke-highlight sekaligus.

Angka 38 itu bukan kebetulan, dan ini inti pelajarannya:

```
TARGET  RGB(228, 32, 196)  ->  H = 155
DECOY   RGB(232, 122, 40)  ->  H =  13

jarak hue melingkar = min(|155-13|, 180-|155-13|) = min(142, 38) = 38
```

Hue itu **melingkar** — 179 nyambung lagi ke 0. Jadi magenta di 155 sebenernya
cuma 38 langkah dari oranye di 13, lewat "belakang" lingkaran. Begitu band lo
selebar itu, ujung atasnya (155+38=193 → balik jadi 13) nabrak decoy pas-pasan.

Jadi `h_tol` aman lo = jarak hue ke warna terdekat yang bukan target. Itu bisa
lo hitung duluan, gak usah nebak. Perhatiin juga metriknya **flat total** dari
h_tol 4 sampai 36 — gak ada bedanya sama sekali — terus jatuh mendadak dari
precision 0.78 ke 0.45. Threshold warna bukan cuma rapuh, dia rapuhnya
**mendadak**: gak ada peringatan pelan-pelan sebelum patah.

Selesai nyetel, ESC — dia bakal cetak flag-nya buat ditempel ke `assist`.

### Tahap 2 — Ukur, jangan cuma ngerasa

Game-nya bisa ngasih label sempurna karena dia sendiri yang naruh targetnya:

```bash
python main.py range --dataset --frames 800
```

Hasilnya `dataset/images/*.png` + `dataset/labels/*.txt` format YOLO. Ini
"ground truth" — kebenaran yang bisa lo pakai buat ngitung seberapa bener
detektor lo, bukan cuma nebak dari lihat layar.

Detektor warna di setelan default, diukur lintas **6 layout pilar acak**:

| seed | precision | recall | F1 | IoU | decoy ketuker |
|---|---|---|---|---|---|
| 1 | 0.800 | 0.800 | 0.800 | 0.768 | 0 |
| 2 | 1.000 | 1.000 | 1.000 | 0.763 | 0 |
| 3 | 0.765 | 0.727 | 0.746 | 0.729 | 0 |
| 4 | 0.975 | 0.950 | 0.962 | 0.781 | 0 |
| 5 | 0.975 | 0.936 | 0.955 | 0.781 | 0 |
| 6 | 1.000 | 1.000 | 1.000 | 0.771 | 0 |
| **mean** | **0.919** | **0.902** | **0.911** | **0.766** | **0** |

Perhatiin sebarannya: **F1 0.746 sampai 1.000**. Detektornya gak berubah sama
sekali — yang berubah cuma di mana pilarnya kebetulan berdiri.

Ini pelajaran kedua, dan mungkin yang paling penting: **kalau lo cuma tes satu
kali, lo bisa dapet angka apa aja di rentang itu** dan ngira itu "hasil" lo.
Selalu ukur di beberapa kondisi, lalu laporkan rata-rata **dan** sebarannya.

Penyebabnya kelihatan jelas di jendela debug: target yang ketutup pilar
blob-nya kepecah jadi dua potongan kecil, masing-masing kebuang `min_area` —
jadi sekaligus satu *miss* (target-nya ilang) dan satu *false positive*
(potongan yang lolos gak match ke bbox mana pun). Itu kelemahan bawaan metode
threshold, dan itulah alasan pindah ke tahap 3.

Yang gak pernah terjadi sekali pun: decoy ketuker jadi target (kolom terakhir
nol semua). Selama band hue-nya masih di dalam jarak aman 38 tadi, pemisahan
kelasnya sempurna.

### Tahap 3 — YOLO

```bash
python main.py train --epochs 40
python main.py assist --detector yolo --weights runs/detect/train/weights/best.pt
```

Bedanya:

|  | Detektor warna | YOLO |
|---|---|---|
| Aturannya | **lo** yang nulis | **dipelajari** dari contoh |
| Butuh dataset | nggak | iya |
| Kecepatan | ~7 ms | ~30–80 ms (CPU) |
| Objek ketutup | gampang gagal | jauh lebih tahan |
| Bedain kelas mirip | rapuh | kuat |
| Debug | gampang (lihat mask) | susah (kotak hitam) |

> Torch di sini versi CPU, jadi training bakal lama. Buat tes cepat dulu:
> `python main.py train --epochs 15 --imgsz 416`.

### Tahap 4 — Utak-atik teori kontrolnya

```bash
python main.py assist --tau 0.03 --fov 400      # agresif, FOV lebar
python main.py assist --tau 0.30 --fov 120      # lembut, FOV sempit
python main.py assist --stickiness 0            # matiin histeresis
python main.py assist --all-classes             # bidik decoy juga
```

`--stickiness 0` itu yang paling ngasih pelajaran: crosshair-nya bakal getar
bolak-balik pas ada dua target berjarak mirip. Itu masalah **asosiasi data** —
detektor gak punya ingatan antar frame, jadi dia gak tau kotak mana yang objek
yang sama kayak frame kemarin. Histeresis itu tambalan paling murahnya;
solusi beneran namanya Kalman filter / Hungarian matching.

---

## Catatan performa

Dari mesin ini (1280×720):

| Bagian | Waktu |
|---|---|
| Deteksi warna | ~7 ms (140 fps) |
| — cvtColor BGR→HSV | ~1.3 ms |
| — inRange + morfologi | ~5 ms |
| Screen capture (mss) | ~17–25 ms |

**Capture-nya yang jadi bottleneck, bukan CV-nya.** mss di Windows mentok
sekitar 60 fps (kena refresh compositor), dan ukuran region hampir gak ngaruh —
grab 256×256 sama cepetnya sama grab 1280×720.

Satu jebakan yang udah dibenerin di `capture.py` dan patut dicatat: jangan
`np.asarray()` objek ScreenShot-nya mss. Dia gak punya buffer protocol, jadi
numpy jatuh ke iterasi per-elemen — itu doang makan ~20 ms per frame 720p,
lebih lama dari capture-nya sendiri. `np.frombuffer(shot.raw, ...)` itu view
zero-copy.

Kalau nanti lo mau lebih cepat dari 60 fps, yang dicari namanya **DXGI Desktop
Duplication** (library `bettercam`/`dxcam`) — bukan optimasi kode CV-nya.

---

## Bagian yang gampang kelewat

**DPI awareness.** Kalau layar lo di-scaling 125%/150%, Windows bakal bohongin
program soal koordinat jendela dan semua kotak lo meleset konsisten.
`enable_dpi_awareness()` harus dipanggil **sebelum** bikin jendela apa pun.

**Frame-rate independence.** `AimController` pakai `alpha = 1 - exp(-dt/tau)`,
bukan konstanta per-frame. Versi naifnya (`step = jarak * 0.3` tiap frame)
bikin aim-nya kerasa beda tiap kali komputer lagi sibuk. Diuji: konvergensi
650 / 617 / 604 ms di 20 / 60 / 144 fps.

**`SetCursorPos` vs `SendInput`.** Yang pertama mindahin penunjuk mouse — jalan
buat game yang baca posisi absolut, termasuk range ini. Yang kedua nyuntikin
gerakan relatif kayak mouse fisik. Dua-duanya ada di `control/mouse.py` biar
kelihatan bedanya.

---

## Ruang lingkup

Lab ini dibangun buat **belajar computer vision pakai sasaran dummy**. Secara
default `assist` cuma nyari jendela practice range yang ada di repo ini.

Kalau lo mau eksperimen lain, arahin ke aplikasi buatan lo sendiri pakai
`--window` atau `--region`. Yang jangan: nempelin ini ke game multiplayer
online. Bukan cuma soal ban — teknik CV-nya sendiri gak jadi lebih menarik di
sana, dan ada orang lain yang mainnya jadi rusak. Semua yang bikin lab ini
berguna buat belajar (ground truth gratis, metrik yang bisa diukur, mask yang
bisa dilihat) justru cuma ada karena target-nya lo yang bikin.

## Kemana lanjutnya

- **Tracking beneran** — ganti histeresis pakai Kalman filter; prediksi posisi
  target berikutnya biar bisa nge-*lead* sasaran yang gerak cepat.
- **Template matching** (`cv2.matchTemplate`) — deteksi berbasis bentuk, bukan warna.
- **Background subtraction** (`cv2.createBackgroundSubtractorMOG2`) — deteksi
  berbasis gerakan, gak peduli warnanya apa.
- **Ekspor ONNX** — `model.export(format="onnx")` bikin inference YOLO di CPU
  jauh lebih cepat.
- **Bikin range-nya lebih susah** — tambah efek cahaya, target semi-transparan,
  background yang warnanya mirip target. Tiap kali lo bikin lebih susah,
  detektor warnanya patah duluan dan YOLO makin kelihatan gunanya.
