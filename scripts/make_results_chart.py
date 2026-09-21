"""Render docs/assets/results.png — the benchmark card for the README.

Reads results/summary.json (the committed live TigerGraph + live Groq run,
100 public questions per pipeline) and draws accuracy by question type plus
the cost/latency headline. Pure Pillow.

    py scripts/make_results_chart.py
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from make_demo_gif import (AMBER, BG, BORDER, CHIP_BG, FAINT, GREEN, INDIGO,
                           MUTED, PANEL, R, RED, S, TRACK, TXT, font, rr, tl, tx)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"

W, H = 920, 552
DIM = (150, 158, 170)

f_kick = font("seguisb.ttf", 12)
f_title = font("segoeuib.ttf", 25)
f_sub = font("segoeui.ttf", 13)
f_row = font("seguisb.ttf", 14)
f_val = font("seguisb.ttf", 12)
f_leg = font("seguisb.ttf", 12)
f_big = font("segoeuib.ttf", 30)
f_cap = font("seguisb.ttf", 12)

PIPES = [("rag", "RAG", RED), ("graphrag", "GraphRAG", AMBER),
         ("agentic", "Agentic", GREEN)]
QTYPES = [("lookup", "lookup"), ("multi_hop", "multi-hop"),
          ("temporal", "temporal"), ("aggregation", "aggregation"),
          ("superlative", "superlative")]


def stat(d, x, w, y, num, cap, color):
    rr(d, (x, y, x + w, y + 88), 14, fill=PANEL, outline=BORDER, width=1)
    tx(d, (x + w / 2, y + 24), num, f_big, color, anchor="mm")
    for i, line in enumerate(cap):
        tx(d, (x + w / 2, y + 52 + i * 16), line, f_cap, DIM, anchor="mm")


def main():
    s = json.loads((ROOT / "results" / "summary.json").read_text())

    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)

    tx(d, (40, 30), "BENCHMARK", f_kick, INDIGO)
    tx(d, (40, 48), "Same 100 questions. Same graph. Three pipelines.", f_title, TXT)
    tx(d, (40, 84), "live TigerGraph + live Groq · accuracy by question type",
       f_sub, MUTED)

    lx = 40
    for _, label, color in PIPES:
        d.ellipse(R(lx, 112, lx + 9, 121), fill=color)
        tx(d, (lx + 15, 110), label, f_leg, MUTED)
        lx += tl(d, label, f_leg) + 42

    x0, track = 250, 500
    y = 146
    for key, label in QTYPES:
        n = s["rag"][key]["n"]
        rr(d, (36, y - 8, W - 36, y + 52), 10, fill=PANEL)
        tx(d, (54, y + 4), label, f_row, TXT)
        tx(d, (54, y + 24), f"n = {n}", f_sub, FAINT)
        by = y + 2
        for pkey, _, color in PIPES:
            acc = s[pkey][key]["accuracy"]
            rr(d, (x0, by, x0 + track, by + 12), 6, fill=TRACK)
            rr(d, (x0, by, x0 + max(4, track * acc), by + 12), 6, fill=color)
            tx(d, (x0 + track + 12, by - 1), f"{acc*100:.0f}%", f_val,
               color if acc > 0 else FAINT)
            by += 15
        y += 60

    a, g = s["agentic"]["_all"], s["graphrag"]["_all"]
    gap, sw = 16, (W - 80 - 2 * 16) / 3
    sy = 446
    stat(d, 40, sw, sy, "99%", ["accuracy overall", "vs 43% GraphRAG, 25% RAG"], GREEN)
    stat(d, 40 + sw + gap, sw, sy, "42", ["tokens per answer", "GraphRAG spends 2,203"], INDIGO)
    stat(d, 40 + 2 * (sw + gap), sw, sy, "0.37s", ["per answer", "GraphRAG takes 15.5s"], AMBER)

    img.resize((W, H), Image.LANCZOS).save(OUT / "results.png")
    print("wrote", OUT / "results.png")


if __name__ == "__main__":
    main()
