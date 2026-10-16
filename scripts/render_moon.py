#!/usr/bin/env python3
"""
Render an 800x480 landscape moon-phase dashboard for the Seeed reTerminal E1002
(7.3" 6-colour E Ink Spectra 6 panel).

Adapted from MostlyFunctionalEngineering/moon_phase_dashboard (1200x1600 portrait
for the E1004) -- astronomy, albedo and dithering code unchanged; layout rebuilt
for the E1002's 800x480 landscape panel: moon on the left, info column on the right.

The image only ever contains the six panel colours (black, white, red, yellow,
green, blue).  The moon itself is shaded in greyscale and error-diffusion
dithered against the panel's measured colours; text, rings and other UI are
drawn as crisp flat colours.

Configuration is via environment variables (all optional):

  TZ_NAME      IANA time zone for dates/times on the display
               (default: Asia/Shanghai)
  LAT, LON     Observer location in decimal degrees (e.g. 34.05 / -118.24).
               Enables moonrise / moonset on the display.
  HEMISPHERE   "north" (default) or "south" -- south flips the moon 180 degrees
               as it appears from the southern hemisphere.
  ROTATE       0 (default), 90, 180 or 270 -- rotate the finished image,
               e.g. to match how the panel is mounted.
  DITHER       "1" (default) to dither the moon, "0" for flat nearest-colour.
  OUTPUT_DIR   Where to write files (default: ./output)
  MOON_NOW     ISO-8601 timestamp to render instead of the current time
               (handy for testing), e.g. 2026-10-26T12:00:00+00:00

Outputs: <OUTPUT_DIR>/moon_dashboard.png and <OUTPUT_DIR>/moon_data.json
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import ephem
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Canvas + palette
# --------------------------------------------------------------------------
W, H = 800, 480
SS = 3  # supersampling factor for the flat-colour UI layer

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / "assets" / "fonts"

# Palette index order is shared by every layer below.
BLACK, WHITE, BLUE, GREEN, RED, YELLOW = range(6)

# Measured colours of an E Ink Spectra 6 panel (what you actually see).
DEVICE = [
    (25, 30, 33),     # black
    (232, 232, 232),  # white
    (33, 87, 186),    # blue
    (18, 95, 32),     # green
    (178, 19, 24),    # red
    (239, 222, 68),   # yellow
]
# Pure primaries -- what most e-paper drivers / converters expect in the file.
PURE = [
    (0, 0, 0),
    (255, 255, 255),
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
]

AU_KM = 149_597_870.7
SYNODIC_MONTH = 29.530588


# --------------------------------------------------------------------------
# Astronomy
# --------------------------------------------------------------------------
def _e(dt):
    return ephem.Date(dt.astimezone(timezone.utc).replace(tzinfo=None))


def _dt(ed):
    return ed.datetime().replace(tzinfo=timezone.utc)


def elongation_deg(dt):
    """Moon-Sun ecliptic elongation, 0=new, 90=first quarter, 180=full, 270=last quarter."""
    m = ephem.Moon(_e(dt))
    return math.degrees(m.elong) % 360.0


def phase_name(e):
    for centre, name in ((0, "New Moon"), (90, "First Quarter"),
                         (180, "Full Moon"), (270, "Last Quarter")):
        if abs((e - centre + 180) % 360 - 180) <= 9:  # roughly +/- 18 h
            return name
    if e < 90:
        return "Waxing Crescent"
    if e < 180:
        return "Waxing Gibbous"
    if e < 270:
        return "Waning Gibbous"
    return "Waning Crescent"


FULL_MOON_NAMES = {
    1: "Wolf Moon", 2: "Snow Moon", 3: "Worm Moon", 4: "Pink Moon",
    5: "Flower Moon", 6: "Strawberry Moon", 7: "Buck Moon", 8: "Sturgeon Moon",
    9: "Corn Moon", 10: "Hunter's Moon", 11: "Beaver Moon", 12: "Cold Moon",
}


def full_moon_name(fm, tz):
    """Traditional name for the full moon at datetime `fm` (UTC)."""
    # Harvest Moon = full moon nearest the September equinox; Hunter's = the next.
    eq = datetime(fm.year, 9, 22, 18, tzinfo=timezone.utc)
    a, b = _dt(ephem.previous_full_moon(_e(eq))), _dt(ephem.next_full_moon(_e(eq)))
    harvest = a if (eq - a) <= (b - eq) else b
    hunters = _dt(ephem.next_full_moon(_e(harvest + timedelta(days=1))))
    if abs((fm - harvest).total_seconds()) < 3600:
        return "Harvest Moon"
    if abs((fm - hunters).total_seconds()) < 3600:
        return "Hunter's Moon"
    local = fm.astimezone(tz)
    prev = _dt(ephem.previous_full_moon(_e(fm - timedelta(hours=1)))).astimezone(tz)
    if (prev.year, prev.month) == (local.year, local.month):
        return "Blue Moon"
    return FULL_MOON_NAMES[local.month]


def rise_set(now, tz, lat, lon):
    """Moonrise / moonset (local datetimes or None) within the local calendar day."""
    obs = ephem.Observer()
    obs.lat, obs.lon = str(lat), str(lon)
    obs.pressure = 0          # geometric horizon; avoids refraction fuzz
    obs.horizon = "-0:34"
    local = now.astimezone(tz)
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    out = {}
    for key, fn in (("rise", obs.next_rising), ("set", obs.next_setting)):
        obs.date = _e(start)
        try:
            t = _dt(fn(ephem.Moon())).astimezone(tz)
            out[key] = t if t < end else None
        except (ephem.NeverUpError, ephem.AlwaysUpError):
            out[key] = None
    return out


def compute_state(now, tz, lat=None, lon=None):
    m = ephem.Moon(_e(now))
    elong = math.degrees(m.elong) % 360.0
    prev_new = _dt(ephem.previous_new_moon(_e(now)))
    nxt = {
        "new": _dt(ephem.next_new_moon(_e(now))),
        "first_quarter": _dt(ephem.next_first_quarter_moon(_e(now))),
        "full": _dt(ephem.next_full_moon(_e(now))),
        "last_quarter": _dt(ephem.next_last_quarter_moon(_e(now))),
    }
    full_dist = ephem.Moon(_e(nxt["full"])).earth_distance * AU_KM
    st = {
        "now": now,
        "elong": elong,
        "illum": m.phase,
        "age": (now - prev_new).total_seconds() / 86400.0,
        "dist_km": m.earth_distance * AU_KM,
        "name": phase_name(elong),
        "next": nxt,
        "full_name": full_moon_name(nxt["full"], tz),
        "full_super": full_dist <= 362_000,
        "rs": None,
    }
    if lat is not None and lon is not None:
        st["rs"] = rise_set(now, tz, lat, lon)
    return st


# --------------------------------------------------------------------------
# Procedural lunar albedo map (no downloads needed)
# --------------------------------------------------------------------------
# (lat, lon, radius_x_km, radius_y_km, rotation_deg)
MARIA = [
    (33, -16, 560, 470, 0),     # Imbrium
    (28, 18, 330, 330, 0),      # Serenitatis
    (8, 31, 430, 300, 25),      # Tranquillitatis
    (17, 59, 240, 170, 0),      # Crisium
    (-5, 52, 330, 450, 20),     # Fecunditatis
    (-15, 34, 170, 170, 0),     # Nectaris
    (-21, -16, 330, 250, 0),    # Nubium
    (-24, -39, 200, 200, 0),    # Humorum
    (20, -55, 650, 780, 0),     # Oceanus Procellarum
    (8, -46, 420, 420, 0),
    (31, -52, 380, 430, 0),
    (-6, -58, 330, 300, 0),
    (-10, -23, 200, 180, 0),    # Cognitum
    (14, 3, 150, 150, 0),       # Vaporum
    (59, -22, 380, 105, 8),     # Frigoris (broken into pieces)
    (61, 12, 300, 90, -6),
    (7, -30, 300, 250, 0),      # Insularum
    (7, 68, 120, 100, 0),       # Undarum
    (-39, 93, 150, 150, 0),
]
# (lat, lon, radius_km, kind)   kind: bright / dark / plain
CRATERS = [
    (-43, -11, 85, "bright"),   # Tycho
    (10, -20, 95, "bright"),    # Copernicus
    (8, -38, 32, "bright"),     # Kepler
    (24, -47, 40, "spot"),      # Aristarchus
    (52, -9, 100, "dark"),      # Plato
    (-6, -68, 110, "dark"),     # Grimaldi
    (-9, 61, 130, "plain"),     # Langrenus
    (16, 47, 28, "spot"),       # Proclus
    (-11, 26, 100, "plain"),    # Theophilus
    (-25, 4, 120, "plain"),
    (-60, -20, 230, "plain"),   # Clavius region
]
MOON_R_KM = 1737.0


def _fbm(size, rng, octaves=((4, 1.0), (8, 0.5), (16, 0.25), (32, 0.13), (64, 0.07), (128, 0.04))):
    out = np.zeros((size, size), np.float32)
    for n, amp in octaves:
        a = rng.random((n, n)).astype(np.float32)
        im = Image.fromarray(a, mode="F").resize((size, size), Image.BICUBIC)
        out += amp * (np.asarray(im, np.float32) - 0.5)
    return out


def _disc_xy(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return math.cos(la) * math.sin(lo), math.sin(la)


def build_albedo(S=1024, seed=11):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:S, 0:S].astype(np.float32)
    x = (xs + 0.5) / S * 2 - 1
    y = 1 - (ys + 0.5) / S * 2

    n_hi, n_edge, n_mar = _fbm(S, rng), _fbm(S, rng), _fbm(S, rng)
    alb = 0.86 + 0.16 * n_hi                      # bright highlands

    mask = np.zeros((S, S), np.float32)
    for lat, lon, rx, ry, rot in MARIA:
        cx, cy = _disc_xy(lat, lon)
        fx = max(0.42, math.cos(math.radians(lon)))
        fy = max(0.42, math.cos(math.radians(lat)))
        a, b = rx / MOON_R_KM * fx, ry / MOON_R_KM * fy
        t = math.radians(rot)
        dx, dy = x - cx, y - cy
        u = dx * math.cos(t) + dy * math.sin(t)
        v = -dx * math.sin(t) + dy * math.cos(t)
        d = np.sqrt((u / a) ** 2 + (v / b) ** 2) * (1 + 0.95 * n_edge) + 0.10 * n_mar
        m = np.clip((1.05 - d) / 0.18, 0, 1)
        mask = np.maximum(mask, m * m * (3 - 2 * m))
    mare = 0.52 + 0.26 * n_mar
    alb = alb * (1 - mask) + mare * mask

    def to_px(cx, cy):
        return (cx + 1) / 2 * S, (1 - cy) / 2 * S

    def crater(px, py, r, depth, rim):
        R = int(r * 2.2) + 2
        x0, x1 = max(0, int(px) - R), min(S, int(px) + R)
        y0, y1 = max(0, int(py) - R), min(S, int(py) + R)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2) / r
        floor = np.where(d < 1, 1 - 0.35 * d, 0)
        rimv = np.exp(-(((d - 1.0) / 0.2) ** 2))
        side = np.clip(0.6 + 0.4 * (xx - px) / (r + 1e-6), 0, 1.4)
        alb[y0:y1, x0:x1] += -depth * floor + rim * rimv * side

    # Random small craters (power-law sizes)
    n = 1100
    ang = rng.random(n) * 2 * math.pi
    rad = np.sqrt(rng.random(n)) * 0.98
    sizes = 0.004 + 0.045 * rng.random(n) ** 3.2
    for i in range(n):
        cx, cy = rad[i] * math.cos(ang[i]), rad[i] * math.sin(ang[i])
        px, py = to_px(cx, cy)
        r = sizes[i] * S / 2
        crater(px, py, r, 0.05 + 0.5 * sizes[i], 0.06 + 0.8 * sizes[i])

    # Named features + ray systems
    for lat, lon, rkm, kind in CRATERS:
        cx, cy = _disc_xy(lat, lon)
        px, py = to_px(cx, cy)
        r = rkm / MOON_R_KM * S / 2 * max(0.5, math.cos(math.radians(lon)))
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
        if kind == "bright":
            alb += 0.22 * np.exp(-((d / (r * 1.1)) ** 2))
            ray_ang = np.arctan2(yy - py, xx - px)
            ray_tab = np.clip(rng.random(720) - 0.62, 0, 1) * 2.6
            ray_tab = np.convolve(np.r_[ray_tab[-3:], ray_tab, ray_tab[:3]], np.ones(5) / 5, "same")[3:-3]
            idx = ((ray_ang + math.pi) / (2 * math.pi) * 719).astype(int)
            alb += 0.28 * ray_tab[idx] * np.exp(-d / (S * 0.18)) * (d > r)
            crater(px, py, r, 0.12, 0.18)
        elif kind == "spot":
            alb += 0.3 * np.exp(-((d / (r * 1.0)) ** 2))
        elif kind == "dark":
            alb -= 0.30 * np.exp(-((d / (r * 0.95)) ** 6))
            crater(px, py, r, 0.0, 0.10)
        else:
            crater(px, py, r, 0.07, 0.14)

    return np.clip(alb, 0.12, 1.0)


# --------------------------------------------------------------------------
# Moon rasteriser (float RGB canvas, dithered afterwards)
# --------------------------------------------------------------------------
class MoonPainter:
    def __init__(self, albedo, south=False):
        self.albedo_img = Image.fromarray((albedo * 255).astype(np.uint8), "L")
        self.south = south
        self._cache = {}
        self.ear = np.zeros((H, W), np.float32)   # probability of a blue earthshine dot

    def _tex(self, D):
        if D not in self._cache:
            im = self.albedo_img.resize((D, D), Image.LANCZOS)
            self._cache[D] = np.asarray(im, np.float32) / 255.0
        return self._cache[D]

    def draw(self, canvas, cx, cy, r, elong_deg):
        R = int(math.ceil(r)) + 2
        D = 2 * R + 1
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
        px, py = xx / r, -yy / r                       # py up
        rr = np.sqrt(px * px + py * py)
        cov = np.clip((1 - rr) * r + 0.5, 0, 1)
        pz = np.sqrt(np.clip(1 - rr * rr, 0, 1))

        e = math.radians(elong_deg)
        lx, lz = math.sin(e), -math.cos(e)             # sun direction (x right, z to viewer)
        mu0 = px * lx + pz * lz
        mu = pz
        lit = np.where(mu0 > 0, 2 * mu0 / (mu0 + mu + 1e-4), 0.0)
        lit = np.clip((1 - np.exp(-1.6 * lit)) / 0.80, 0, 1)

        # texture lookup (disc-plane coords -> albedo map)
        tex = self._tex(D)
        ix = np.clip(((px + 1) / 2 * D).astype(int), 0, D - 1)
        iy = np.clip(((1 - py) / 2 * D).astype(int), 0, D - 1)
        alb = tex[iy, ix]

        v = lit * alb
        # faint earthshine on the dark side, rendered later as sparse blue stipple
        ear = (1 - np.minimum(1, lit * 4)) * (0.010 + 0.040 * alb) * cov

        black = np.array(DEVICE[BLACK], np.float32)
        white = np.array(DEVICE[WHITE], np.float32)
        rgb = black + v[..., None] * (white - black)

        if self.south:
            rgb = rgb[::-1, ::-1]
            cov = cov[::-1, ::-1]
            ear = ear[::-1, ::-1]

        x0, y0 = cx - R, cy - R
        sx0, sy0 = max(0, -x0), max(0, -y0)
        sx1, sy1 = min(D, W - x0), min(D, H - y0)
        if sx0 >= sx1 or sy0 >= sy1:
            return
        dst = canvas[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1]
        c = cov[sy0:sy1, sx0:sx1, None]
        dst[:] = dst * (1 - c) + rgb[sy0:sy1, sx0:sx1] * c
        e = self.ear[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1]
        np.maximum(e, ear[sy0:sy1, sx0:sx1], out=e)


# --------------------------------------------------------------------------
# Palette quantisation helpers
# --------------------------------------------------------------------------
def _palette_image(indices):
    flat = []
    for i in indices:
        flat += list(DEVICE[i])
    flat += list(DEVICE[indices[0]]) * (256 - len(indices))  # pad with duplicates, never new colours
    p = Image.new("P", (1, 1))
    p.putpalette(flat)
    return p


def quantise(rgb_img, dither, indices=(BLACK, WHITE)):
    """Quantise to a subset of the panel colours; returns indices in the global palette.

    The moon is shaded in grey, so by default it is dithered with black and white
    only -- letting red/green/blue join in turns greys into coloured speckle.
    """
    q = rgb_img.quantize(
        palette=_palette_image(indices),
        dither=Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE,
    )
    local = np.asarray(q, np.uint8)
    lut = np.array(list(indices) + [indices[0]] * (256 - len(indices)), np.uint8)
    return lut[local]


# --------------------------------------------------------------------------
# Flat-colour UI layer (indexed, supersampled, majority-resolved)
# --------------------------------------------------------------------------
TRANSPARENT = 255


class Ui:
    def __init__(self):
        self.im = Image.new("P", (W * SS, H * SS), TRANSPARENT)
        pal = []
        for c in DEVICE:
            pal += list(c)
        pal += [255, 0, 255] * (256 - len(DEVICE))
        self.im.putpalette(pal)
        self.d = ImageDraw.Draw(self.im)
        self._fonts = {}

    def font(self, weight, size):
        key = (weight, size)
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype(str(FONT_DIR / f"Inter-{weight}.otf"), int(round(size * SS)))
        return self._fonts[key]

    # -- primitives (all coordinates in 1x pixels) --
    def circle(self, cx, cy, r, fill=None, outline=None, width=1):
        self.d.ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS],
                       fill=fill, outline=outline, width=int(round(width * SS)))

    def arc(self, cx, cy, r, a0, a1, color, width):
        self.d.arc([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS],
                   a0, a1, fill=color, width=int(round(width * SS)))

    def pie(self, cx, cy, r, a0, a1, color):
        self.d.pieslice([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS],
                        a0, a1, fill=color)

    def rect(self, x0, y0, x1, y1, color):
        self.d.rectangle([x0 * SS, y0 * SS, x1 * SS - 1, y1 * SS - 1], fill=color)

    def text_width(self, s, weight, size, tracking=0):
        f = self.font(weight, size)
        if tracking:
            return (sum(f.getlength(ch) for ch in s) + tracking * SS * (len(s) - 1)) / SS
        return f.getlength(s) / SS

    def text(self, x, y, s, weight, size, color, align="l", tracking=0):
        """y is the text baseline."""
        f = self.font(weight, size)
        w = self.text_width(s, weight, size, tracking)
        x0 = x if align == "l" else x - w / 2 if align == "m" else x - w
        if not tracking:
            self.d.text((x0 * SS, y * SS), s, font=f, fill=color, anchor="ls")
            return w
        cx = x0 * SS
        for ch in s:
            self.d.text((cx, y * SS), ch, font=f, fill=color, anchor="ls")
            cx += f.getlength(ch) + tracking * SS
        return w

    def sparkle(self, cx, cy, r, color):
        """Four-point star."""
        pts = []
        for i in range(8):
            a = i * math.pi / 4
            rr = r if i % 2 == 0 else r * 0.22
            pts.append(((cx + rr * math.cos(a)) * SS, (cy + rr * math.sin(a)) * SS))
        self.d.polygon(pts, fill=color)

    def resolve(self):
        """Collapse the supersampled index map to (H, W) indices + coverage mask."""
        a = np.asarray(self.im, np.uint8).reshape(H, SS, W, SS)
        counts = np.stack([(a == k).sum(axis=(1, 3)) for k in range(len(DEVICE))], axis=0)
        best = counts.argmax(axis=0).astype(np.uint8)
        total = counts.sum(axis=0)
        return best, total >= (SS * SS + 1) // 2


# --------------------------------------------------------------------------
# Layout (800x480 landscape: moon left, info column right)
# --------------------------------------------------------------------------
CX, CY = 190, 278          # moon centre
R_MOON = 132
R_RING = 162

X0 = 380                   # left edge of the right info column
COL_W = W - 28 - X0        # info column width (392 px)


def fmt_time(dt):
    h = dt.hour % 12 or 12
    return f"{h}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def fmt_date(dt):
    return f"{dt:%b} {dt.day}"


def rel(target, now):
    h = (target - now).total_seconds() / 3600
    if h < 1:
        return "in under an hour"
    if h < 48:
        return f"in {round(h)} hours"
    return f"in {round(h / 24)} days"


def ring_pos(angle_deg, r=R_RING):
    a = math.radians(-90 + angle_deg)
    return CX + r * math.cos(a), CY + r * math.sin(a)


def draw_stars(ui):
    import random
    rng = random.Random(42)
    placed = []
    tries = 0
    while len(placed) < 36 and tries < 5000:
        tries += 1
        x, y = rng.uniform(16, 356), rng.uniform(84, 464)
        if math.hypot(x - CX, y - CY) < R_RING + 20:
            continue
        if any(math.hypot(x - a, y - b) < 24 for a, b, _ in placed):
            continue
        placed.append((x, y, rng.random()))
    for x, y, t in placed:
        if t > 0.975:
            ui.circle(x, y, 2.6, fill=RED)
        elif t > 0.955:
            ui.circle(x, y, 2.3, fill=GREEN)
        elif t > 0.93:
            ui.circle(x, y, 2.1, fill=YELLOW)
        elif t > 0.86:
            ui.circle(x, y, 1.9, fill=BLUE)
        elif t > 0.5:
            ui.circle(x, y, 1.7, fill=WHITE)
        else:
            ui.circle(x, y, 1.1, fill=WHITE)
    for x, y, c in ((58, 104, YELLOW), (322, 100, WHITE), (330, 424, YELLOW), (44, 428, WHITE)):
        ui.sparkle(x, y, 11, c)


def draw_ring(ui, st):
    e = st["elong"]
    ui.circle(CX, CY, R_RING, outline=BLUE, width=2)
    ui.arc(CX, CY, R_RING, -90, -90 + e, YELLOW, 6)
    # four phase markers
    for ang, kind in ((0, "new"), (90, "fq"), (180, "full"), (270, "lq")):
        x, y = ring_pos(ang)
        ui.circle(x, y, 13, fill=BLACK)
        if kind == "new":
            ui.circle(x, y, 9, fill=BLACK, outline=BLUE, width=2)
        elif kind == "full":
            ui.circle(x, y, 9, fill=WHITE)
        else:
            ui.circle(x, y, 9, fill=BLACK, outline=WHITE, width=2)
            ui.pie(x, y, 9, -90, 90, WHITE) if kind == "fq" else ui.pie(x, y, 9, 90, 270, WHITE)
    # current position
    x, y = ring_pos(e)
    ui.circle(x, y, 17, fill=BLACK)
    ui.circle(x, y, 12, fill=YELLOW)
    ui.circle(x, y, 4, fill=BLACK)


def draw_header(ui, st, tz):
    local = st["now"].astimezone(tz)
    ui.rect(28, 26, 34, 32, YELLOW)
    ui.text(44, 48, "LUNAR PHASE", "SemiBold", 18, WHITE, tracking=4)
    ui.text(W - 28, 48, f"{local:%a}  {local:%b} {local.day}, {local.year}".upper(),
            "SemiBold", 18, YELLOW, align="r", tracking=2)


def draw_title(ui, st):
    name = st["name"]
    size = 46
    while ui.text_width(name, "SemiBold", size) > COL_W and size > 18:
        size -= 2
    ui.text(X0, 138, name, "SemiBold", size, WHITE)
    illum = int(round(st["illum"]))
    ui.text(X0, 170, f"{illum}% illuminated   ·   day {st['age']:.1f} of 29.5",
            "Medium", 19, WHITE)

    dist = f"{st['dist_km']:,.0f} km away"
    if st["dist_km"] < 364_000:
        dist += "  ·  near perigee"
    elif st["dist_km"] > 404_000:
        dist += "  ·  near apogee"
    ui.text(X0, 200, dist, "Medium", 16, YELLOW)

    rs = st["rs"]
    if rs:
        bits = []
        if rs["rise"]:
            bits.append(f"rises {fmt_time(rs['rise'])}")
        if rs["set"]:
            bits.append(f"sets {fmt_time(rs['set'])}")
        if bits:
            ui.text(X0, 224, "   ·   ".join(bits), "Medium", 15, WHITE)


def draw_cards(ui, st, tz):
    now = st["now"]
    cards = [
        (246, "NEXT NEW MOON", st["next"]["new"], BLUE, None),
        (322, "NEXT FULL MOON", st["next"]["full"], YELLOW,
         st["full_name"].replace(" Moon", " Supermoon") if st["full_super"] else st["full_name"]),
    ]
    for top, label, when, accent, extra in cards:
        loc = when.astimezone(tz)
        ui.rect(X0, top, X0 + 6, top + 66, accent)
        tx = X0 + 24
        ui.text(tx, top + 20, label, "SemiBold", 14, WHITE, tracking=2)
        ui.text(tx, top + 52, fmt_date(loc), "SemiBold", 32, WHITE)
        if extra:
            ui.text(X0 + COL_W, top + 52, extra, "Medium", 15, YELLOW, align="r")
        ui.text(tx, top + 64, f"{loc:%a} {fmt_time(loc)}   ·   {rel(when, now)}", "Medium", 14, WHITE)


def draw_week_labels(ui, days, tz):
    y_line = 404
    ui.rect(X0, y_line, X0 + COL_W, y_line + 2, BLUE)
    for i, (centre_x, dt) in enumerate(days):
        loc = dt.astimezone(tz)
        label = "TODAY" if i == 0 else f"{loc:%a} {loc.day}".upper()
        ui.text(centre_x, 424, label, "SemiBold", 11, YELLOW if i == 0 else WHITE, align="m", tracking=1)
        if i == 0:
            ui.circle(centre_x, WEEK_Y, 20, outline=YELLOW, width=2)


WEEK_Y = 444
WEEK_R = 14


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    tz = ZoneInfo(os.environ.get("TZ_NAME") or "Asia/Shanghai")
    now_s = os.environ.get("MOON_NOW")
    now = datetime.fromisoformat(now_s) if now_s else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)

    lat = os.environ.get("LAT")
    lon = os.environ.get("LON")
    lat = float(lat) if lat else None
    lon = float(lon) if lon else None
    south = (os.environ.get("HEMISPHERE") or "north").strip().lower().startswith("s")
    dither = (os.environ.get("DITHER") or "1") != "0"
    rotate = int(os.environ.get("ROTATE") or 0)
    out_dir = Path(os.environ.get("OUTPUT_DIR") or "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    st = compute_state(now, tz, lat, lon)

    # ---- moons (big + next 7 days strip) ----
    canvas = np.zeros((H, W, 3), np.float32)
    canvas[:] = DEVICE[BLACK]
    painter = MoonPainter(build_albedo(), south=south)
    painter.draw(canvas, CX, CY, R_MOON, st["elong"])

    local_now = now.astimezone(tz)
    days = []
    slot = COL_W / 7
    for i in range(7):
        if i == 0:
            when = now
        else:
            noon = (local_now + timedelta(days=i)).replace(hour=12, minute=0, second=0, microsecond=0)
            when = noon.astimezone(timezone.utc)
        x = int(round(X0 + slot * (i + 0.5)))
        painter.draw(canvas, x, WEEK_Y, WEEK_R, elongation_deg(when))
        days.append((x, when))

    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")
    idx = quantise(img, dither)
    rng = np.random.default_rng(5)
    stipple = (rng.random((H, W)) < painter.ear) & (idx == BLACK)
    idx[stipple] = BLUE

    # ---- flat UI layer ----
    ui = Ui()
    draw_stars(ui)
    draw_ring(ui, st)
    draw_header(ui, st, tz)
    draw_title(ui, st)
    draw_cards(ui, st, tz)
    draw_week_labels(ui, days, tz)
    ui_idx, ui_mask = ui.resolve()
    idx = np.where(ui_mask, ui_idx, idx)

    lut = np.array(PURE if (os.environ.get("OUTPUT_PALETTE") or "pure") == "pure" else DEVICE, np.uint8)
    out = Image.fromarray(lut[idx], "RGB")
    if rotate:
        out = out.rotate(-rotate, expand=True)  # clockwise
    out.save(out_dir / "moon_dashboard.png", optimize=True)

    iso = lambda d: d.isoformat().replace("+00:00", "Z")
    data = {
        "generated_at": iso(now),
        "phase": st["name"],
        "illumination_pct": round(st["illum"], 1),
        "age_days": round(st["age"], 2),
        "elongation_deg": round(st["elong"], 1),
        "distance_km": round(st["dist_km"]),
        "next_new_moon": iso(st["next"]["new"]),
        "next_first_quarter": iso(st["next"]["first_quarter"]),
        "next_full_moon": iso(st["next"]["full"]),
        "next_last_quarter": iso(st["next"]["last_quarter"]),
        "next_full_moon_name": st["full_name"],
        "next_full_moon_is_supermoon": bool(st["full_super"]),
        "moonrise": iso(st["rs"]["rise"].astimezone(timezone.utc)) if st["rs"] and st["rs"]["rise"] else None,
        "moonset": iso(st["rs"]["set"].astimezone(timezone.utc)) if st["rs"] and st["rs"]["set"] else None,
    }
    (out_dir / "moon_data.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"{st['name']} - {st['illum']:.0f}% - wrote {out_dir / 'moon_dashboard.png'}")


if __name__ == "__main__":
    main()
