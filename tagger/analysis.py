"""Results by tag, with small samples held back.

The performance numbers are synthetic: a seeded generator gives each sample ad
impressions and clicks. They exist to show the analysis, not to report results.
"""

from __future__ import annotations

import random

from .taxonomy import FIELDS

MIN_ADS = 3            # a tag value seen on fewer ads than this is reported but flagged
PRIOR_IMPRESSIONS = 20000  # strength of the pull toward the overall click rate


def synthetic_performance(briefs: list[dict], seed: int = 7) -> dict:
    """Invented impressions and clicks per ad, with one planted effect (people shown)."""
    rng = random.Random(seed)
    rows = {}
    for brief in briefs:
        impressions = rng.randrange(8000, 40000, 500)
        rate = 0.011
        if brief["intended"]["people"] in ("one_person", "small_group"):
            rate *= 1.35
        rate *= rng.uniform(0.85, 1.15)
        rows[brief["id"]] = {"impressions": impressions, "clicks": round(impressions * rate)}
    return rows


def by_tag(tag_run: dict, performance: dict) -> dict:
    total_imp = sum(p["impressions"] for p in performance.values())
    total_clk = sum(p["clicks"] for p in performance.values())
    base = total_clk / total_imp
    table = []
    for field in FIELDS:
        groups: dict[str, dict] = {}
        for image_id, runs in tag_run["images"].items():
            if not runs or image_id not in performance:
                continue
            group = groups.setdefault(runs[0][field]["value"],
                                      {"ads": 0, "impressions": 0, "clicks": 0})
            group["ads"] += 1
            group["impressions"] += performance[image_id]["impressions"]
            group["clicks"] += performance[image_id]["clicks"]
        for value, g in sorted(groups.items()):
            shrunk = (g["clicks"] + base * PRIOR_IMPRESSIONS) / (g["impressions"] + PRIOR_IMPRESSIONS)
            table.append({
                "field": field, "value": value, "ads": g["ads"],
                "impressions": g["impressions"], "clicks": g["clicks"],
                "ctr": round(g["clicks"] / g["impressions"], 5),
                "shrunk_lift": round(shrunk / base, 3),
                "enough_data": g["ads"] >= MIN_ADS,
            })
    return {"overall_ctr": round(base, 5), "min_ads": MIN_ADS,
            "prior_impressions": PRIOR_IMPRESSIONS, "rows": table}
