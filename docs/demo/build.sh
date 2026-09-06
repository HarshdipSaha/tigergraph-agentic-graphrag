#!/usr/bin/env bash
# Rebuilds final-demo.mp4 from the slide HTML sources.
# Prereqs: ffmpeg on PATH, Python with pywin32 (Windows SAPI TTS), and a running
# Playwright-controlled browser to screenshot the slides (see step 2 — this repo
# built the frames via the Claude Code Playwright MCP tools; a plain
# `playwright screenshot` CLI call works identically if you have the playwright
# package installed: `pip install playwright && playwright install chromium`).
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p frames audio scenes

echo "1. Generating narration audio (offline SAPI TTS)..."
python tts.py

echo "2. Render each slides/sceneN.html -> frames/sceneN.png at 1920x1080 yourself"
echo "   (via Playwright, or: playwright screenshot --viewport-size=1920,1080 slides/sceneN.html frames/sceneN.png)"
echo "   Skipping if frames/ already populated."

echo "3. Compositing each scene (image + audio + fade)..."
for s in scene1 scene2 scene3 scene4 scene5 scene6; do
  dur=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "audio/$s.wav")
  total=$(python -c "print(round(float('$dur')+0.6,2))")
  fadeout_start=$(python -c "print(round(float('$total')-0.4,2))")
  ffmpeg -y -loop 1 -i "frames/$s.png" -i "audio/$s.wav" \
    -vf "fade=t=in:st=0:d=0.3,fade=t=out:st=${fadeout_start}:d=0.4" \
    -af "adelay=300|300,apad" \
    -c:v libx264 -tune stillimage -pix_fmt yuv420p -r 30 \
    -c:a aac -b:a 192k \
    -t "$total" \
    "scenes/$s.mp4" -loglevel error
done

echo "4. Concatenating..."
printf "file 'scenes/scene1.mp4'\nfile 'scenes/scene2.mp4'\nfile 'scenes/scene3.mp4'\nfile 'scenes/scene4.mp4'\nfile 'scenes/scene5.mp4'\nfile 'scenes/scene6.mp4'\n" > concat.txt
ffmpeg -y -f concat -safe 0 -i concat.txt -c copy final-demo.mp4 -loglevel error

echo "Done: docs/demo/final-demo.mp4"
