"""Draw the tags beside each ad, and build a contact sheet for the README."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .taxonomy import FIELDS

PANEL = 420
INK, MUTED, BG, GOOD, WARN = (232, 237, 242), (154, 167, 181), (18, 22, 28), (95, 196, 156), (240, 168, 80)


def annotate(image_bytes: bytes, tags: dict, intended: dict | None = None, size: int = 640) -> Image.Image:
    """The ad on the left, its tags with confidence bars on the right."""
    ad = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((size, size))
    canvas = Image.new("RGB", (size + PANEL, size), BG)
    canvas.paste(ad, (0, 0))
    draw = ImageDraw.Draw(canvas)
    label, value_font = ImageFont.load_default(size=15), ImageFont.load_default(size=21)
    draw.text((size + 24, 20), "VISION TAGS", font=label, fill=MUTED)
    y = 54
    step = (size - 70) // len(FIELDS)
    for field in FIELDS:
        entry = tags[field]
        match = intended is None or intended.get(field) == entry["value"]
        draw.text((size + 24, y), field.replace("_", " "), font=label, fill=MUTED)
        draw.text((size + 24, y + 19), entry["value"].replace("_", " "), font=value_font,
                  fill=INK if match else WARN)
        bar_x, bar_w = size + 250, PANEL - 280
        draw.rectangle((bar_x, y + 28, bar_x + bar_w, y + 36), fill=(43, 52, 64))
        draw.rectangle((bar_x, y + 28, bar_x + int(bar_w * entry["confidence"]), y + 36),
                       fill=GOOD if match else WARN)
        draw.text((bar_x, y + 4), f"{entry['confidence']:.0%}", font=label, fill=MUTED)
        y += step
    return canvas


def contact_sheet(panels: list[Image.Image], columns: int = 2, gap: int = 16) -> Image.Image:
    width, height = panels[0].size
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width + (columns + 1) * gap,
                              rows * height + (rows + 1) * gap), (8, 11, 15))
    for index, panel in enumerate(panels):
        x = gap + (index % columns) * (width + gap)
        y = gap + (index // columns) * (height + gap)
        sheet.paste(panel, (x, y))
    return sheet


def write_all(ads_dir: Path, tag_run: dict, briefs: list[dict], out_dir: Path, sheet_ids: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    intended = {b["id"]: b["intended"] for b in briefs}
    panels = {}
    for image_id, runs in tag_run["images"].items():
        if not runs:
            continue
        panel = annotate((ads_dir / f"{image_id}.png").read_bytes(), runs[0], intended.get(image_id))
        panel.save(out_dir / f"{image_id}.jpg", quality=85)
        panels[image_id] = panel
    chosen = [panels[i] for i in sheet_ids if i in panels]
    if chosen:
        contact_sheet(chosen).save(out_dir / "contact-sheet.jpg", quality=85)
