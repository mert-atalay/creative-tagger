"""Two checks on the tagger: does it recover what the brief asked for, and is it stable?"""

from __future__ import annotations

from .taxonomy import FIELDS


def agreement(tag_run: dict, briefs: list[dict]) -> dict:
    """First-pass tags compared with the attributes each image was made to have."""
    intended = {b["id"]: b["intended"] for b in briefs}
    per_field = {f: {"match": 0, "total": 0, "misses": []} for f in FIELDS}
    for image_id, runs in tag_run["images"].items():
        if not runs or image_id not in intended:
            continue
        for field in FIELDS:
            got, want = runs[0][field]["value"], intended[image_id][field]
            row = per_field[field]
            row["total"] += 1
            if got == want:
                row["match"] += 1
            else:
                row["misses"].append({"id": image_id, "intended": want, "tagged": got,
                                      "confidence": runs[0][field]["confidence"]})
    for row in per_field.values():
        row["rate"] = round(row["match"] / row["total"], 3) if row["total"] else None
    matched = sum(r["match"] for r in per_field.values())
    total = sum(r["total"] for r in per_field.values())
    return {"overall": round(matched / total, 3) if total else None,
            "matched": matched, "total": total, "per_field": per_field}


def stability(tag_run: dict) -> dict:
    """Share of fields where every pass on the same image gave the same value."""
    per_field = {f: {"same": 0, "total": 0} for f in FIELDS}
    for runs in tag_run["images"].values():
        if len(runs) < 2:
            continue
        for field in FIELDS:
            per_field[field]["total"] += 1
            if len({r[field]["value"] for r in runs}) == 1:
                per_field[field]["same"] += 1
    for row in per_field.values():
        row["rate"] = round(row["same"] / row["total"], 3) if row["total"] else None
    same = sum(r["same"] for r in per_field.values())
    total = sum(r["total"] for r in per_field.values())
    return {"overall": round(same / total, 3) if total else None, "per_field": per_field}
