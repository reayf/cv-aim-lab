"""
Practice Range -- game tembak-tembakan offline buat jadi bahan latihan CV.

Ini "dunia" buat lab-nya. Dia gambar target yang gerak, decoy pengecoh, dan
pilar yang nutupin sebagian target. Semuanya offline, single player, gak ada
lawan manusia -- murni sasaran dummy.

Mode dataset (--dataset) nyimpen tiap frame ke-N sebagai PNG plus label YOLO.
Karena game-nya sendiri tau persis di mana targetnya, label-nya gratis dan
akurat. Ini cara paling cepat dapet dataset ber-anotasi buat belajar.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


# --------------------------------------------------------------------------
# Entity
# --------------------------------------------------------------------------
class Entity:
    """Bola yang gerak. kind 0 = target (boleh ditembak), 1 = decoy."""

    __slots__ = ("x", "y", "r", "vx", "vy", "kind", "phase", "bob")

    def __init__(self, play, kind):
        self.kind = kind
        self.respawn(play)

    def respawn(self, play):
        self.r = random.uniform(C.RADIUS_MIN, C.RADIUS_MAX)
        self.x = random.uniform(play.left + self.r, play.right - self.r)
        self.y = random.uniform(play.top + self.r, play.bottom - self.r)
        angle = random.uniform(0, math.tau)
        speed = random.uniform(C.SPEED_MIN, C.SPEED_MAX)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.phase = random.uniform(0, math.tau)
        self.bob = random.uniform(0.0, 26.0)

    def update(self, dt, play):
        self.phase += dt * 2.2
        self.x += self.vx * dt
        self.y += self.vy * dt + math.cos(self.phase) * self.bob * dt

        # pantul di dalam area main
        if self.x - self.r < play.left:
            self.x, self.vx = play.left + self.r, abs(self.vx)
        elif self.x + self.r > play.right:
            self.x, self.vx = play.right - self.r, -abs(self.vx)
        if self.y - self.r < play.top:
            self.y, self.vy = play.top + self.r, abs(self.vy)
        elif self.y + self.r > play.bottom:
            self.y, self.vy = play.bottom - self.r, -abs(self.vy)

    @property
    def bbox(self):
        """(left, top, w, h) dalam pixel layar game."""
        return (self.x - self.r, self.y - self.r, self.r * 2, self.r * 2)

    def draw(self, surf):
        fill = C.TARGET_RGB if self.kind == C.CLS_TARGET else C.DECOY_RGB
        ring = C.TARGET_RING_RGB if self.kind == C.CLS_TARGET else C.DECOY_RING_RGB
        pos = (int(self.x), int(self.y))
        pygame.draw.circle(surf, fill, pos, int(self.r))
        pygame.draw.circle(surf, ring, pos, int(self.r), max(2, int(self.r * 0.12)))
        pygame.draw.circle(surf, ring, pos, max(2, int(self.r * 0.16)))


# --------------------------------------------------------------------------
# Marker hit/miss
# --------------------------------------------------------------------------
class Marker:
    __slots__ = ("x", "y", "t", "hit")
    LIFE = 0.35

    def __init__(self, x, y, hit):
        self.x, self.y, self.hit, self.t = x, y, hit, 0.0

    def update(self, dt):
        self.t += dt
        return self.t < self.LIFE

    def draw(self, surf):
        k = 1.0 - self.t / self.LIFE
        col = C.HIT_RGB if self.hit else C.MISS_RGB
        size = int(6 + 18 * (1 - k))
        w = max(1, int(3 * k))
        x, y = int(self.x), int(self.y)
        pygame.draw.line(surf, col, (x - size, y - size), (x + size, y + size), w)
        pygame.draw.line(surf, col, (x - size, y + size), (x + size, y - size), w)


# --------------------------------------------------------------------------
# Range
# --------------------------------------------------------------------------
class PracticeRange:
    def __init__(self, dataset_dir=None, every=3, max_frames=1500, seed=None):
        if seed is not None:
            random.seed(seed)

        pygame.init()
        pygame.display.set_caption(C.WINDOW_TITLE)
        self.screen = pygame.display.set_mode((C.GAME_W, C.GAME_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 19)
        self.font_big = pygame.font.SysFont("consolas", 30, bold=True)

        self.play = pygame.Rect(
            C.PLAY_MARGIN_X,
            C.PLAY_TOP,
            C.GAME_W - 2 * C.PLAY_MARGIN_X,
            C.GAME_H - C.PLAY_TOP - C.PLAY_BOTTOM,
        )

        self.targets = [Entity(self.play, C.CLS_TARGET) for _ in range(C.N_TARGETS)]
        self.decoys = [Entity(self.play, C.CLS_DECOY) for _ in range(C.N_DECOYS)]
        self.pillars = self._make_pillars()
        self.markers = []

        self.shots = 0
        self.hits = 0
        self.penalty = 0          # nembak decoy
        self.start = time.perf_counter()
        self.paused = False

        # state mode dataset
        self.dataset_dir = dataset_dir
        self.every = max(1, every)
        self.max_frames = max_frames
        self.frame_i = 0
        self.saved = 0
        if dataset_dir:
            os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, "labels"), exist_ok=True)
            with open(os.path.join(dataset_dir, "classes.txt"), "w") as f:
                f.write("\n".join(C.CLASS_NAMES) + "\n")

    def _make_pillars(self):
        rects = []
        slot = self.play.width // (C.N_PILLARS + 1)
        for i in range(1, C.N_PILLARS + 1):
            w = random.randint(34, 62)
            h = random.randint(int(self.play.height * 0.45), int(self.play.height * 0.85))
            x = self.play.left + slot * i - w // 2 + random.randint(-40, 40)
            y = self.play.top + random.randint(0, max(1, self.play.height - h))
            rects.append(pygame.Rect(x, y, w, h))
        return rects

    # ---------------- occlusion ----------------
    def covered_fraction(self, ent):
        """Perkiraan kasar berapa bagian bbox entity yang ketutup pilar."""
        bx, by, bw, bh = ent.bbox
        box = pygame.Rect(int(bx), int(by), int(bw), int(bh))
        area = max(1, box.width * box.height)
        covered = 0
        for p in self.pillars:
            clip = box.clip(p)
            covered += clip.width * clip.height
        return min(1.0, covered / area)

    # ---------------- input ----------------
    def shoot(self, mx, my):
        self.shots += 1

        # pilar itu solid: nembak pilar = meleset
        for p in self.pillars:
            if p.collidepoint(mx, my):
                self.markers.append(Marker(mx, my, False))
                return

        for ent in self.targets + self.decoys:
            if math.hypot(mx - ent.x, my - ent.y) <= ent.r:
                if ent.kind == C.CLS_TARGET:
                    self.hits += 1
                    self.markers.append(Marker(mx, my, True))
                    ent.respawn(self.play)
                else:
                    self.penalty += 1
                    self.markers.append(Marker(mx, my, False))
                return

        self.markers.append(Marker(mx, my, False))

    # ---------------- loop ----------------
    def run(self):
        running = True
        while running:
            dt = min(self.clock.tick(C.GAME_FPS) / 1000.0, 0.05)

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif ev.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif ev.key == pygame.K_r:
                        self.shots = self.hits = self.penalty = 0
                        self.start = time.perf_counter()
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self.shoot(*ev.pos)

            if not self.paused:
                for ent in self.targets + self.decoys:
                    ent.update(dt, self.play)
            self.markers = [m for m in self.markers if m.update(dt)]

            self.draw()
            pygame.display.flip()

            if self.dataset_dir:
                self.frame_i += 1
                if self.frame_i % self.every == 0:
                    self.save_frame()
                    if self.saved >= self.max_frames:
                        print(f"\n[dataset] selesai: {self.saved} frame -> {self.dataset_dir}")
                        running = False

        pygame.quit()
        self.report()

    # ---------------- render ----------------
    def draw(self):
        s = self.screen
        s.fill(C.BG_RGB)

        # grid lantai -> kasih tekstur ke background biar deteksi gak terlalu gampang
        for gx in range(self.play.left, self.play.right, 64):
            pygame.draw.line(s, C.GRID_RGB, (gx, self.play.top), (gx, self.play.bottom))
        for gy in range(self.play.top, self.play.bottom, 64):
            pygame.draw.line(s, C.GRID_RGB, (self.play.left, gy), (self.play.right, gy))
        pygame.draw.rect(s, C.GRID_RGB, self.play, 2)

        for ent in self.decoys:
            ent.draw(s)
        for ent in self.targets:
            ent.draw(s)

        # pilar digambar SETELAH entity -> jadi target bisa ketutup sebagian
        for p in self.pillars:
            pygame.draw.rect(s, C.PILLAR_RGB, p)
            pygame.draw.rect(s, C.PILLAR_EDGE_RGB, p, 2)

        for m in self.markers:
            m.draw(s)

        self.draw_crosshair()
        self.draw_hud()

    def draw_crosshair(self):
        mx, my = pygame.mouse.get_pos()
        c = C.CROSSHAIR_RGB
        pygame.draw.line(self.screen, c, (mx - 12, my), (mx - 4, my), 2)
        pygame.draw.line(self.screen, c, (mx + 4, my), (mx + 12, my), 2)
        pygame.draw.line(self.screen, c, (mx, my - 12), (mx, my - 4), 2)
        pygame.draw.line(self.screen, c, (mx, my + 4), (mx, my + 12), 2)

    def draw_hud(self):
        acc = (self.hits / self.shots * 100) if self.shots else 0.0
        elapsed = time.perf_counter() - self.start
        left = (
            f"HIT {self.hits:3d}   SHOT {self.shots:3d}   ACC {acc:5.1f}%   "
            f"DECOY {self.penalty:2d}   {elapsed:5.1f}s"
        )
        self.screen.blit(self.font_big.render(left, True, C.HUD_RGB), (C.PLAY_MARGIN_X, 22))

        hint = "LMB tembak magenta  |  jangan kena oranye  |  SPACE pause  R reset  ESC keluar"
        if self.dataset_dir:
            hint = f"[MODE DATASET] {self.saved}/{self.max_frames} frame tersimpan  |  ESC stop"
        self.screen.blit(
            self.font.render(hint, True, C.HUD_DIM_RGB),
            (C.PLAY_MARGIN_X, C.GAME_H - 30),
        )
        fps = self.font.render(f"{self.clock.get_fps():4.0f} fps", True, C.HUD_DIM_RGB)
        self.screen.blit(fps, (C.GAME_W - C.PLAY_MARGIN_X - fps.get_width(), C.GAME_H - 30))

    # ---------------- dataset ----------------
    def save_frame(self):
        """Simpan frame + label YOLO. Ini ground truth yang gratis dan sempurna."""
        lines = []
        for ent in self.targets + self.decoys:
            # skip yang hampir sepenuhnya ketutup pilar -- itu label sampah
            if self.covered_fraction(ent) > 0.75:
                continue
            bx, by, bw, bh = ent.bbox
            cx = (bx + bw / 2) / C.GAME_W
            cy = (by + bh / 2) / C.GAME_H
            nw, nh = bw / C.GAME_W, bh / C.GAME_H
            if not (0 < cx < 1 and 0 < cy < 1):
                continue
            lines.append(f"{ent.kind} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        name = f"frame_{self.saved:05d}"
        pygame.image.save(self.screen, os.path.join(self.dataset_dir, "images", name + ".png"))
        with open(os.path.join(self.dataset_dir, "labels", name + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        self.saved += 1
        if self.saved % 50 == 0:
            print(f"[dataset] {self.saved}/{self.max_frames}", end="\r", flush=True)

    def report(self):
        if self.dataset_dir:
            return
        acc = (self.hits / self.shots * 100) if self.shots else 0.0
        print(f"\nHit {self.hits} / Shot {self.shots}  ->  akurasi {acc:.1f}%  "
              f"(decoy kena {self.penalty}x)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="CV Aim Lab -- practice range")
    ap.add_argument("--dataset", metavar="DIR", nargs="?", const="dataset", default=None,
                    help="rekam frame + label YOLO ke DIR (default: ./dataset)")
    ap.add_argument("--every", type=int, default=3, help="simpan tiap N frame (default 3)")
    ap.add_argument("--frames", type=int, default=1500, help="berhenti setelah N frame tersimpan")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)

    PracticeRange(a.dataset, a.every, a.frames, a.seed).run()


if __name__ == "__main__":
    main()
