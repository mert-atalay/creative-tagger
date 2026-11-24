"""The tag schema. One place defines every field and its allowed values."""

from __future__ import annotations

TAXONOMY: dict[str, dict] = {
    "people": {
        "question": "Who is shown?",
        "values": ["none", "one_person", "small_group", "crowd"],
    },
    "setting": {
        "question": "Where does the scene take place?",
        "values": ["gym_floor", "studio_class", "outdoors", "home", "abstract_or_plain"],
    },
    "activity": {
        "question": "What is the main activity?",
        "values": ["strength", "cardio", "class", "stretch_or_recovery", "coaching", "none"],
    },
    "mood": {
        "question": "What is the overall mood?",
        "values": ["calm", "energetic", "friendly", "intense"],
    },
    "shot": {
        "question": "How is the subject framed?",
        "values": ["close_up", "medium", "wide", "no_subject"],
    },
    "dominant_colour": {
        "question": "Which colour family dominates the image?",
        "values": ["warm", "cool", "neutral", "dark", "bright_mixed"],
    },
    "text_amount": {
        "question": ("How much overlaid text is there? Do not count the brand name beside "
                     "the logo or the words on a button."),
        "values": ["none", "headline_only", "headline_and_more"],
    },
    "cta_present": {
        "question": "Is there a visible call-to-action button?",
        "values": ["yes", "no"],
    },
}

FIELDS = list(TAXONOMY)


def response_schema() -> dict:
    """JSON schema the vision model must follow: a value and a confidence per field."""
    props = {}
    for name, spec in TAXONOMY.items():
        props[name] = {
            "type": "object",
            "properties": {
                "value": {"type": "string", "enum": spec["values"]},
                "confidence": {"type": "number"},
            },
            "required": ["value", "confidence"],
        }
    return {"type": "object", "properties": props, "required": FIELDS}


def prompt() -> str:
    lines = [
        "You are tagging one advertising image against a fixed taxonomy.",
        "Answer every field with one of its allowed values and a confidence from 0 to 1.",
        "Judge only what is visible. Do not guess brand, identity, age or any personal trait.",
        "",
    ]
    for name, spec in TAXONOMY.items():
        lines.append(f"- {name}: {spec['question']} Allowed: {', '.join(spec['values'])}.")
    return "\n".join(lines)


def validate(tags: dict) -> list[str]:
    """Return a list of problems; empty means the tags fit the taxonomy."""
    problems = []
    for name, spec in TAXONOMY.items():
        entry = tags.get(name)
        if not isinstance(entry, dict):
            problems.append(f"{name}: missing")
            continue
        if entry.get("value") not in spec["values"]:
            problems.append(f"{name}: value {entry.get('value')!r} is not allowed")
        conf = entry.get("confidence")
        if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
            problems.append(f"{name}: confidence must be between 0 and 1")
    extra = set(tags) - set(TAXONOMY)
    if extra:
        problems.append(f"unknown fields: {sorted(extra)}")
    return problems
