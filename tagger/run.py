"""Tag every ad image, one or more passes each."""

from __future__ import annotations

from pathlib import Path

from . import taxonomy


def tag_all(client, model: str, ads_dir: Path, passes: int = 1) -> dict:
    """Return {"model", "passes", "images": {id: [tags_pass_1, ...]}, "invalid": [...]}"""
    prompt, schema = taxonomy.prompt(), taxonomy.response_schema()
    images, invalid = {}, []
    for path in sorted(ads_dir.glob("*.png")):
        runs = []
        for number in range(passes):
            tags = client.tag_image(model, path.read_bytes(), prompt, schema)
            problems = taxonomy.validate(tags)
            if problems:
                invalid.append({"id": path.stem, "pass": number + 1, "problems": problems})
                continue
            runs.append(tags)
        images[path.stem] = runs
    return {"model": model, "passes": passes, "images": images, "invalid": invalid}
