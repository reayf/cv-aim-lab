#!/usr/bin/env python
"""
CV Aim Lab -- satu pintu masuk buat semua bagian lab.

    python main.py range              main practice range-nya
    python main.py range --dataset    main sambil rekam dataset berlabel
    python main.py assist             jalanin loop CV + aim assist
    python main.py train              latih YOLO dari dataset rekaman
    python main.py tune               setel band warna secara interaktif

Jalanin `python main.py <perintah> --help` buat opsi lengkap tiap perintah.
"""
from __future__ import annotations

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0

    cmd, rest = argv[0], argv[1:]

    if cmd == "range":
        from range_game.target_range import main as run
        return run(rest)

    if cmd == "assist":
        from control.assist import main as run
        return run(rest)

    if cmd == "train":
        from tools.train_yolo import main as run
        return run(rest)

    if cmd == "tune":
        from tools.tune_color import main as run
        return run(rest)

    print(f"perintah gak dikenal: {cmd!r}")
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
