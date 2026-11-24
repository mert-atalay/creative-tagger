"""Make the sample ads: a generated background plus text and a button placed by code.

The image model draws the scene only. The headline, the button and the brand
mark are composited afterwards, so brand elements are never model-drawn.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = 1024
BRAND = "Harbour Fitness"
ORANGE = (255, 107, 53)
STYLE = ("Photorealistic advertising photograph, square format. No text, no letters, "
         "no logos, no watermarks anywhere in the image. Scene: ")


def load_briefs(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    return lines + [current] if current else lines


def compose(background: bytes, brief: dict) -> bytes:
    """Place the brand mark, headline, optional sub-line and optional button."""
    image = Image.open(io.BytesIO(background)).convert("RGB").resize((SIZE, SIZE))
    intended = brief["intended"]
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse((40, 40, 84, 84), fill=ORANGE + (255,))
    draw.text((98, 46), BRAND, font=_font(30), fill=(255, 255, 255, 255),
              stroke_width=2, stroke_fill=(0, 0, 0, 160))
    if intended["text_amount"] != "none":
        draw.rectangle((0, SIZE - 330, SIZE, SIZE), fill=(8, 20, 34, 170))
        y = SIZE - 300
        for line in _wrap(draw, brief["headline"], _font(64), SIZE - 120)[:2]:
            draw.text((60, y), line, font=_font(64), fill=(255, 255, 255, 255))
            y += 76
        if intended["text_amount"] == "headline_and_more":
            draw.text((60, y + 6), "First week free. Terms apply.", font=_font(30),
                      fill=(230, 236, 242, 255))
    if intended["cta_present"] == "yes":
        draw.rounded_rectangle((60, SIZE - 110, 300, SIZE - 46), radius=32, fill=ORANGE + (255,))
        draw.text((104, SIZE - 96), "Book now", font=_font(32), fill=(255, 255, 255, 255))
    out = io.BytesIO()
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(out, "PNG")
    return out.getvalue()


def build(client, model: str, briefs: list[dict], raw_dir: Path, ads_dir: Path) -> int:
    """Generate missing backgrounds (one call each, no retries) and compose every ad."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    ads_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    for brief in briefs:
        raw = raw_dir / f"{brief['id']}.png"
        if not raw.exists():
            raw.write_bytes(client.generate_image(model, STYLE + brief["scene"]))
            generated += 1
        (ads_dir / f"{brief['id']}.png").write_bytes(compose(raw.read_bytes(), brief))
    return generated
