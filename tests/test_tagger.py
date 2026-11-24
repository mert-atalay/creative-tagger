import io
import json
from pathlib import Path

import pytest
from PIL import Image

from tagger import analysis, annotate, quality, run, samples, taxonomy
from tagger.client import BudgetExceeded, GeminiClient

ROOT = Path(__file__).resolve().parent.parent
BRIEFS = samples.load_briefs(ROOT / "briefs" / "briefs.json")


def png(colour=(40, 80, 120)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 64), colour).save(out, "PNG")
    return out.getvalue()


def tags_for(intended: dict, confidence: float = 0.9) -> dict:
    return {f: {"value": intended[f], "confidence": confidence} for f in taxonomy.FIELDS}


class FakeClient:
    """Returns the intended tags, except where `wrong` overrides a field for an image."""

    def __init__(self, wrong=None, flip_second_pass=None):
        self.wrong, self.flip, self.seen, self.images = wrong or {}, flip_second_pass or {}, {}, 0

    def generate_image(self, model, prompt):
        self.images += 1
        return png()

    def tag_image(self, model, image_bytes, prompt, schema):
        image_id = self.current
        count = self.seen[image_id] = self.seen.get(image_id, 0) + 1
        tags = tags_for(next(b["intended"] for b in BRIEFS if b["id"] == image_id))
        for field, value in self.wrong.get(image_id, {}).items():
            tags[field]["value"] = value
        if count == 2:
            for field, value in self.flip.get(image_id, {}).items():
                tags[field]["value"] = value
        return tags


def tag_run(tmp_path, client, passes=2):
    """Build the sample ads with the fake client and tag each one `passes` times."""
    ads = tmp_path / "ads"
    samples.build(client, "image-model", BRIEFS, tmp_path / "raw", ads)
    result = {"model": "m", "passes": passes, "images": {}, "invalid": []}
    for path in sorted(ads.glob("*.png")):
        client.current = path.stem
        result["images"][path.stem] = [
            client.tag_image("m", path.read_bytes(), "", {}) for _ in range(passes)]
    return result, ads


def test_briefs_fit_the_taxonomy():
    assert len(BRIEFS) == 12
    for brief in BRIEFS:
        assert taxonomy.validate(tags_for(brief["intended"])) == []


def test_validate_rejects_bad_tags():
    good = tags_for(BRIEFS[0]["intended"])
    bad_value = json.loads(json.dumps(good)); bad_value["people"]["value"] = "robots"
    bad_conf = json.loads(json.dumps(good)); bad_conf["mood"]["confidence"] = 1.4
    missing = {k: v for k, v in good.items() if k != "shot"}
    extra = dict(good, age_group={"value": "x", "confidence": 1})
    assert any("people" in p for p in taxonomy.validate(bad_value))
    assert any("mood" in p for p in taxonomy.validate(bad_conf))
    assert any("shot" in p for p in taxonomy.validate(missing))
    assert any("unknown" in p for p in taxonomy.validate(extra))


def test_prompt_and_schema_cover_every_field():
    schema = taxonomy.response_schema()
    for field, spec in taxonomy.TAXONOMY.items():
        assert field in taxonomy.prompt()
        assert schema["properties"][field]["properties"]["value"]["enum"] == spec["values"]
    assert "personal trait" in taxonomy.prompt()


def test_samples_generate_once_and_compose(tmp_path):
    client = FakeClient()
    assert samples.build(client, "m", BRIEFS, tmp_path / "raw", tmp_path / "ads") == 12
    assert samples.build(client, "m", BRIEFS, tmp_path / "raw", tmp_path / "ads") == 0
    assert client.images == 12
    ad = Image.open(tmp_path / "ads" / "a01.png")
    assert ad.size == (samples.SIZE, samples.SIZE)
    plain = Image.open(io.BytesIO(samples.compose(png(), BRIEFS[6])))   # a07: no text, no button
    with_text = Image.open(io.BytesIO(samples.compose(png(), BRIEFS[0])))
    assert plain.getpixel((100, 950)) != with_text.getpixel((100, 950))


def test_tag_all_drops_invalid_responses(tmp_path):
    ads = tmp_path / "ads"; ads.mkdir()
    (ads / "a01.png").write_bytes(png())

    class Bad:
        def tag_image(self, *args):
            return {"people": {"value": "robots", "confidence": 2}}

    result = run.tag_all(Bad(), "m", ads, passes=1)
    assert result["images"]["a01"] == [] and len(result["invalid"]) == 1


def test_agreement_counts_misses(tmp_path):
    result, _ = tag_run(tmp_path, FakeClient(wrong={"a03": {"mood": "energetic"}}))
    report = quality.agreement(result, BRIEFS)
    assert report["total"] == 12 * len(taxonomy.FIELDS)
    assert report["matched"] == report["total"] - 1
    assert report["per_field"]["mood"]["misses"][0]["id"] == "a03"


def test_stability_detects_a_changed_answer(tmp_path):
    steady, _ = tag_run(tmp_path / "s", FakeClient())
    assert quality.stability(steady)["overall"] == 1.0
    shaky, _ = tag_run(tmp_path / "u", FakeClient(flip_second_pass={"a05": {"shot": "wide"}}))
    report = quality.stability(shaky)
    assert report["per_field"]["shot"]["same"] == 11 and report["overall"] < 1


def test_analysis_is_deterministic_and_flags_small_samples(tmp_path):
    result, _ = tag_run(tmp_path, FakeClient())
    perf = analysis.synthetic_performance(BRIEFS)
    assert perf == analysis.synthetic_performance(BRIEFS)
    table = analysis.by_tag(result, perf)
    rows = {(r["field"], r["value"]): r for r in table["rows"]}
    assert rows[("people", "crowd")]["ads"] == 1 and not rows[("people", "crowd")]["enough_data"]
    assert rows[("people", "one_person")]["enough_data"]
    assert sum(r["ads"] for r in table["rows"] if r["field"] == "people") == 12
    # the planted effect: people in the picture beats none, and shrinkage keeps lifts modest
    assert rows[("people", "one_person")]["shrunk_lift"] > rows[("people", "none")]["shrunk_lift"]
    assert all(0.6 < r["shrunk_lift"] < 1.6 for r in table["rows"])


def test_annotated_images_and_contact_sheet(tmp_path):
    result, ads = tag_run(tmp_path, FakeClient(wrong={"a01": {"mood": "calm"}}))
    annotate.write_all(ads, result, BRIEFS, tmp_path / "annotated", ["a01", "a02", "a03", "a06"])
    panel = Image.open(tmp_path / "annotated" / "a01.jpg")
    assert panel.size == (640 + annotate.PANEL, 640)
    assert (tmp_path / "annotated" / "contact-sheet.jpg").exists()


def test_client_budget_and_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiClient(max_calls=1)
    client = GeminiClient(max_calls=0, api_key="test-key")
    with pytest.raises(BudgetExceeded):
        client.tag_image("m", png(), "p", {})
    assert client.calls == 0


@pytest.mark.parametrize("name", ["run1", "run2"])
def test_committed_reports_match_their_tags(name):
    """The numbers quoted in the README come from these files and must reproduce."""
    folder = ROOT / "results" / name
    tags = json.loads((folder / "tags.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    assert quality.agreement(tags, BRIEFS) == report["agreement_with_briefs"]
    assert quality.stability(tags) == report["stability_across_passes"]
    assert analysis.by_tag(tags, analysis.synthetic_performance(BRIEFS)) == report["by_tag"]
