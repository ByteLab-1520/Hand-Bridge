"""
Generate Hand-Bridge GitHub profile image.
Run: python gen_profile.py
Output: profile.png
"""

import os
import math
from PIL import Image, ImageDraw, ImageFilter

W, H = 512, 512

# ── 60-30-10 Color Rule ───────────────────────────────────────────────────────
# 60% dominant  → deep dark navy
# 30% secondary → dark slate blue  (robot hand, panels)
# 10% accent    → neon green       (pose lines, dots, highlights)

# 60% ── dominant
BG          = (11,  13,  20,  255)   # #0b0d14
GRID        = (18,  22,  34,  255)   # subtle grid on BG
SCAN        = (0,   255, 136, 6)     # scan lines — 10% accent, very faint

# 30% ── secondary  (robot hand body)
HAND_BODY   = (28,  35,  51,  255)   # #1c2333
HAND_SHINE  = (46,  60,  85,  255)   # #2e3c55  lighter panel
HAND_EDGE   = (15,  18,  28,  255)   # #0f121c  dark edge
JOINT_OUT   = (36,  48,  68,  255)   # #243044  surface
JOINT_IN    = (58,  80,  112, 255)   # #3a5070  lighter surface
BOLT        = (180, 200, 220, 255)   # near-white bolt highlight
PANEL_LINE  = (20,  27,  44,  255)   # subtle inner detail

# 10% ── accent  (all pose estimation, UI pops)
GREEN       = (0,   255, 136)        # #00ff88  main accent
GREEN_DIM   = (0,   180, 96)         # dimmed accent
WHITE       = (255, 255, 255)        # neutral — wrist dot only
BRACKET_C   = (0,   255, 136, 210)
DIM_TEXT    = (0,   200, 100, 180)

# ── Hand landmark positions (open palm, fingers up, 512×512) ──────────────────
LM = [
    (256, 435),  # 0  Wrist
    (302, 394),  # 1  Thumb CMC
    (340, 350),  # 2  Thumb MCP
    (364, 308),  # 3  Thumb IP
    (382, 268),  # 4  Thumb TIP ★
    (318, 332),  # 5  Index MCP
    (320, 268),  # 6  Index PIP
    (320, 215),  # 7  Index DIP
    (320, 165),  # 8  Index TIP ★
    (260, 325),  # 9  Middle MCP
    (260, 258),  # 10 Middle PIP
    (260, 200),  # 11 Middle DIP
    (258, 145),  # 12 Middle TIP ★
    (202, 330),  # 13 Ring MCP
    (200, 268),  # 14 Ring PIP
    (198, 215),  # 15 Ring DIP
    (197, 168),  # 16 Ring TIP ★
    (153, 348),  # 17 Pinky MCP
    (148, 295),  # 18 Pinky PIP
    (144, 252),  # 19 Pinky DIP
    (142, 215),  # 20 Pinky TIP ★
]

CONNECTIONS = [
    (0, 1),(1, 2),(2, 3),(3, 4),           # thumb
    (0, 5),(5, 6),(6, 7),(7, 8),           # index
    (0, 9),(9,10),(10,11),(11,12),         # middle
    (0,13),(13,14),(14,15),(15,16),        # ring
    (0,17),(17,18),(18,19),(19,20),        # pinky
    (5, 9),(9,13),(13,17),                 # palm arch
]

TIPS    = {4, 8, 12, 16, 20}
PALM_PTS = {0, 5, 9, 13, 17}


# ── Helpers ───────────────────────────────────────────────────────────────────

def circle(draw, cx, cy, r, fill, outline=None, ow=1):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 fill=fill, outline=outline, width=ow)


def thick_line(draw, p1, p2, color, width):
    draw.line([p1, p2], fill=color, width=width)


def perp_offset(p1, p2, half_w):
    """Unit vector perpendicular to (p1→p2), scaled by half_w."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy) or 1
    return (-dy / length * half_w, dx / length * half_w)


def bone_rect(draw, p1, p2, half_w, fill, outline):
    """Draw a filled rectangle (bone segment) between p1 and p2."""
    ox, oy = perp_offset(p1, p2, half_w)
    pts = [
        (p1[0] + ox, p1[1] + oy),
        (p2[0] + ox, p2[1] + oy),
        (p2[0] - ox, p2[1] - oy),
        (p1[0] - ox, p1[1] - oy),
    ]
    draw.polygon(pts, fill=fill, outline=outline)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    img  = Image.new('RGBA', (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Background grid
    for x in range(0, W, 20):
        draw.line([(x, 0), (x, H)], fill=GRID, width=1)
    for y in range(0, H, 20):
        draw.line([(0, y), (W, y)], fill=GRID, width=1)

    # Scan lines
    for y in range(0, H, 3):
        draw.line([(0, y), (W, y)], fill=SCAN, width=1)

    # ── Robot hand body ───────────────────────────────────────────────────────

    # Palm filled polygon
    palm_pts = [LM[0], LM[1], LM[5], LM[9], LM[13], LM[17]]
    draw.polygon(palm_pts, fill=HAND_BODY, outline=HAND_EDGE)

    # Panel lines on palm (robotic detail)
    cx = sum(p[0] for p in palm_pts) // len(palm_pts)
    cy = sum(p[1] for p in palm_pts) // len(palm_pts)
    for pt in palm_pts:
        mid = ((pt[0] + cx) // 2, (pt[1] + cy) // 2)
        draw.line([mid, (cx, cy)], fill=PANEL_LINE, width=1)

    # Bone segments — wider at base, narrower at tip
    finger_chains = [
        ([1, 2, 3, 4],    [11, 9, 8]),    # thumb
        ([5, 6, 7, 8],    [12, 10, 8]),   # index
        ([9,10,11,12],    [13, 11, 8]),   # middle
        ([13,14,15,16],   [12, 10, 8]),   # ring
        ([17,18,19,20],   [10,  8, 7]),   # pinky
    ]

    for chain, widths in finger_chains:
        for seg_i in range(len(chain) - 1):
            a, b  = LM[chain[seg_i]], LM[chain[seg_i + 1]]
            hw    = widths[min(seg_i, len(widths)-1)]
            bone_rect(draw, a, b, hw, HAND_BODY, HAND_EDGE)
            # Shine stripe
            ox, oy = perp_offset(a, b, hw * 0.25)
            draw.line(
                [(a[0]+ox, a[1]+oy), (b[0]+ox, b[1]+oy)],
                fill=HAND_SHINE, width=2,
            )

    # Fingertip caps
    for idx in TIPS:
        x, y = LM[idx]
        r = 10 if idx == 4 else 9
        circle(draw, x, y, r, HAND_SHINE, HAND_EDGE, 2)
        circle(draw, x, y, 4, BOLT)

    # Joint circles
    for idx, (x, y) in enumerate(LM):
        if idx in TIPS:
            continue
        r = 10 if idx == 0 else 7
        circle(draw, x, y, r, JOINT_OUT, HAND_EDGE, 2)
        circle(draw, x, y, r - 3, JOINT_IN)
        circle(draw, x, y, 2, BOLT)

    # ── Pose estimation overlay ───────────────────────────────────────────────

    # Glow pass — draw onto separate layer then blur
    glow_layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow_layer)
    for a_i, b_i in CONNECTIONS:
        gd.line([LM[a_i], LM[b_i]], fill=GREEN + (80,), width=10)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=5))
    img = Image.alpha_composite(img, glow_layer)
    draw = ImageDraw.Draw(img)

    # Core lines
    for a_i, b_i in CONNECTIONS:
        draw.line([LM[a_i], LM[b_i]], fill=GREEN + (220,), width=2)

    # Landmark dots
    for idx, (x, y) in enumerate(LM):
        if idx in TIPS:
            col, r, inner = WHITE,     8, 4   # fingertips — bright white pop (10% accent)
        elif idx in PALM_PTS:
            col, r, inner = GREEN,     7, 3   # palm — main accent
        else:
            col, r, inner = GREEN_DIM, 6, 3   # mid joints — dimmed accent
        circle(draw, x, y, r, col + (255,), (0, 0, 0, 200), 1)
        circle(draw, x, y, inner, WHITE + (255,))

    # ── Corner brackets ───────────────────────────────────────────────────────
    m, L = 18, 30
    for (cx, cy), (sx, sy) in [
        ((m, m),     (1, 1)),
        ((W-m, m),   (-1, 1)),
        ((m, H-m),   (1, -1)),
        ((W-m, H-m), (-1, -1)),
    ]:
        draw.line([(cx, cy), (cx + sx*L, cy)],         fill=BRACKET_C, width=2)
        draw.line([(cx, cy), (cx, cy + sy*L)],         fill=BRACKET_C, width=2)
        draw.line([(cx + sx*4, cy + sy*4),
                   (cx + sx*(L-4), cy + sy*4)],        fill=BRACKET_C, width=1)

    # ── Title ─────────────────────────────────────────────────────────────────
    from PIL import ImageFont
    try:
        sys_mono = {
            "Darwin":  "/System/Library/Fonts/Courier.dfont",
            "Windows": "C:/Windows/Fonts/cour.ttf",
            "Linux":   "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        }
        import platform
        fpath = sys_mono.get(platform.system(), "")
        font_big = ImageFont.truetype(fpath, 20) if os.path.exists(fpath) else ImageFont.load_default()
        font_sm  = ImageFont.truetype(fpath, 11) if os.path.exists(fpath) else ImageFont.load_default()
    except Exception:
        font_big = ImageFont.load_default()
        font_sm  = font_big

    # Dim background strip for text
    draw.rectangle([(0, H - 52), (W, H)], fill=(10, 12, 18, 200))
    draw.line([(0, H - 52), (W, H - 52)], fill=GREEN + (80,), width=1)

    draw.text((W // 2, H - 36), "HAND-BRIDGE",
              font=font_big, fill=GREEN + (230,), anchor="mm")
    draw.text((W // 2, H - 16),
              "Real-time Korean Sign Language Translator",
              font=font_sm, fill=GREEN_DIM + (180,), anchor="mm")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profile.png')
    img.save(out)
    print(f"Saved → {out}")


if __name__ == '__main__':
    main()
