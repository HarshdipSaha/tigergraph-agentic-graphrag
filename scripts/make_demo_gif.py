"""Render docs/assets/demo.gif — one real benchmark question, three pipelines.

Everything shown is the committed result for pub-045 (results/*_public.jsonl):
a counting question where top-k retrieval physically cannot see the whole
answer set, and the certificate proves the agentic pipeline did.

Pure Pillow + Windows system fonts. Run:
    py scripts/make_demo_gif.py            # writes docs/assets/demo.gif
    py scripts/make_demo_gif.py --still    # writes docs/assets/demo_still.png
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

BG      = (13, 16, 22)
PANEL   = (22, 27, 34)
BORDER  = (44, 51, 62)
TRACK   = (33, 39, 48)
TXT     = (230, 237, 243)
MUTED   = (139, 148, 158)
FAINT   = (85, 95, 108)
AMBER   = (240, 185, 80)
RED     = (248, 96, 88)
GREEN   = (63, 185, 80)
INDIGO  = (150, 130, 250)
CHIP_BG = (32, 38, 47)

S = 2
W, H = 920, 404

F = "C:/Windows/Fonts/"
def font(name, size):
    return ImageFont.truetype(F + name, size * S)

f_lbl  = font("seguisb.ttf", 11)
f_q    = font("seguisb.ttf", 17)
f_chip = font("seguisb.ttf", 13)
f_note = font("segoeui.ttf", 13)
f_ans  = font("segoeuib.ttf", 26)
f_sub  = font("segoeui.ttf", 12)
f_foot = font("seguisb.ttf", 13)

GOLD = 20
TOTAL = 43

ROWS = [
    dict(name="RAG", color=RED, seen=4, answer="1",
         note="top-4 chunks by similarity", ok=False, cost="1,236 tokens"),
    dict(name="GraphRAG", color=AMBER, seen=12, answer="2",
         note="top-k + one-hop expansion", ok=False, cost="2,123 tokens"),
    dict(name="Agentic + certificate", color=GREEN, seen=43, answer="20",
         note="graph COUNT = 43, all 43 inspected", ok=True, cost="0 tokens"),
]


def R(*v):
    return tuple(int(round(x * S)) for x in v)


def rr(d, box, radius, fill=None, outline=None, width=1):
    d.rounded_rectangle(R(*box), radius=int(radius * S), fill=fill,
                        outline=outline, width=int(width * S))


def tx(d, xy, s, fnt, fill, anchor="la"):
    d.text((int(xy[0] * S), int(xy[1] * S)), s, font=fnt, fill=fill, anchor=anchor)


def tl(d, s, fnt):
    return d.textlength(s, font=fnt) / S


def mark(d, cx, cy, r, ok, color):
    d.ellipse(R(cx - r, cy - r, cx + r, cy + r), fill=color)
    if ok:
        pts = [R(cx - .42 * r, cy + .02 * r), R(cx - .12 * r, cy + .34 * r),
               R(cx + .44 * r, cy - .32 * r)]
        d.line([p for xy in pts for p in xy], fill=BG, width=int(2.3 * S), joint="curve")
    else:
        o = r * 0.40
        for a, b in (((-o, -o), (o, o)), ((-o, o), (o, -o))):
            d.line(R(cx + a[0], cy + a[1]) + R(cx + b[0], cy + b[1]),
                   fill=BG, width=int(2.3 * S))


def render(step):
    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)

    rr(d, (22, 22, W - 22, 104), 14, fill=PANEL, outline=BORDER, width=1)
    tx(d, (42, 34), "BENCHMARK QUESTION  ·  pub-045  ·  gold answer: 20", f_lbl, MUTED)
    tx(d, (42, 53), "How many athletics events at the 2004 Summer Olympics", f_q, TXT)
    tx(d, (42, 75), "had more than 41 competitors?", f_q, TXT)

    y = 120
    for i, row in enumerate(ROWS):
        if step < i + 1:
            break
        rr(d, (22, y, W - 22, y + 66), 12, fill=PANEL, outline=BORDER, width=1)
        rr(d, (22, y, 27, y + 66), 3, fill=row["color"])

        cw = tl(d, row["name"], f_chip)
        rr(d, (42, y + 12, 42 + cw + 22, y + 36), 12, fill=CHIP_BG)
        tx(d, (53, y + 16), row["name"], f_chip, row["color"])
        tx(d, (42, y + 44), row["note"], f_sub, FAINT)

        # evidence coverage bar
        bx, bw = 300, 330
        rr(d, (bx, y + 17, bx + bw, y + 29), 6, fill=TRACK)
        fw = max(5, bw * row["seen"] / TOTAL)
        rr(d, (bx, y + 17, bx + fw, y + 29), 6, fill=row["color"])
        tx(d, (bx, y + 38), f"saw {row['seen']} of {TOTAL} matching events",
           f_note, MUTED)
        tx(d, (bx + bw + 14, y + 16), row["cost"], f_sub, FAINT)

        mark(d, W - 150, y + 33, 10, row["ok"], row["color"])
        tx(d, (W - 46, y + 17), row["answer"], f_ans, row["color"], anchor="ra")
        tx(d, (W - 46, y + 46), "answered", f_sub, FAINT, anchor="ra")
        y += 76

    if step >= 4:
        tx(d, (W / 2, 356),
           "The certificate checks evidence set == the graph's own COUNT.",
           f_foot, TXT, anchor="ma")
        tx(d, (W / 2, 378),
           "Top-k retrieval cannot know what it never retrieved.",
           f_sub, FAINT, anchor="ma")

    return img


def build_frames():
    frames, durs = [], []

    def add(im, ms):
        frames.append(im); durs.append(ms)

    def xfade(a, b, n=3, ms=42):
        for i in range(1, n + 1):
            add(Image.blend(a, b, i / (n + 1)), ms)

    holds = {0: 950, 1: 1350, 2: 1350, 3: 1700, 4: 2800}
    prev = None
    for step in range(5):
        cur = render(step)
        if prev is not None:
            xfade(prev, cur)
        add(cur, holds[step])
        prev = cur
    return frames, durs


def down(im):
    return im.resize((W, H), Image.LANCZOS)


def main():
    if "--still" in sys.argv:
        down(render(4)).save(OUT / "demo_still.png")
        print("wrote", OUT / "demo_still.png")
        return

    frames, durs = build_frames()
    frames = [down(f) for f in frames]
    master = Image.new("RGB", (W, H * 2), BG)
    master.paste(frames[-1], (0, 0)); master.paste(frames[0], (0, H))
    pal = master.quantize(colors=220, method=Image.MEDIANCUT, dither=Image.NONE)
    pf = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]

    p = OUT / "demo.gif"
    pf[0].save(p, save_all=True, append_images=pf[1:], loop=0, duration=durs,
               disposal=2, optimize=True)
    print(f"wrote {p}  ({len(pf)} frames, {p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
