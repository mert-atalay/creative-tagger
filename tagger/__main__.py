"""Command line: python -m tagger <samples|tag|report>

samples  generate missing backgrounds and compose the sample ads (uses the image model)
tag      tag every ad with the vision model and write out/tags.json
report   quality checks, by-tag analysis and annotated images from out/tags.json (no API calls)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import analysis, annotate, quality, run, samples

ROOT = Path(__file__).resolve().parent.parent
VISION_MODEL = os.environ.get("TAGGER_VISION_MODEL", "gemini-flash-latest")
IMAGE_MODEL = os.environ.get("TAGGER_IMAGE_MODEL", "gemini-2.5-flash-image")
SHEET = ["a01", "a02", "a03", "a06"]


def main() -> None:
    parser = argparse.ArgumentParser(prog="tagger", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["samples", "tag", "report"])
    parser.add_argument("--passes", type=int, default=2, help="tagging passes per image")
    parser.add_argument("--max-calls", type=int, default=30, help="hard limit on API calls")
    parser.add_argument("--out", default=str(ROOT / "out"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    briefs = samples.load_briefs(ROOT / "briefs" / "briefs.json")
    ads_dir = ROOT / "samples" / "ads"

    if args.command in ("samples", "tag"):
        from .client import GeminiClient, OpenAIImageClient, OpenAIVisionClient
        if args.command == "samples":
            # The image model can come from either provider; the name decides.
            maker = OpenAIImageClient if IMAGE_MODEL.startswith("gpt-") else GeminiClient
            client = maker(max_calls=args.max_calls)
            made = samples.build(client, IMAGE_MODEL, briefs, ROOT / "samples" / "raw", ads_dir)
            print(f"generated {made} backgrounds with {IMAGE_MODEL}, composed {len(briefs)} ads")
        else:
            tagger_class = OpenAIVisionClient if VISION_MODEL.startswith("gpt-") else GeminiClient
            client = tagger_class(max_calls=args.max_calls)
            result = run.tag_all(client, VISION_MODEL, ads_dir, args.passes)
            (out / "tags.json").write_text(json.dumps(result, indent=2))
            print(f"tagged {len(result['images'])} images, {args.passes} passes, "
                  f"{client.calls} API calls, {len(result['invalid'])} invalid responses")
        return

    tag_run = json.loads((out / "tags.json").read_text())
    report = {
        "model": tag_run["model"],
        "agreement_with_briefs": quality.agreement(tag_run, briefs),
        "stability_across_passes": quality.stability(tag_run),
        "by_tag": analysis.by_tag(tag_run, analysis.synthetic_performance(briefs)),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    annotate.write_all(ads_dir, tag_run, briefs, out / "annotated", SHEET)
    agree, stable = report["agreement_with_briefs"], report["stability_across_passes"]
    print(f"agreement with briefs: {agree['matched']}/{agree['total']} ({agree['overall']:.0%})")
    print(f"stability across passes: {stable['overall']:.0%}" if stable["overall"] is not None
          else "stability: needs at least two passes")
    for field, row in agree["per_field"].items():
        print(f"  {field:<16} {row['match']}/{row['total']}")


if __name__ == "__main__":
    main()
