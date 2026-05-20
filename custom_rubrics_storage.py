import json
import os
import re

CUSTOM_RUBRICS_FILE = "custom_rubrics.json"

def get_raw_custom_rubrics() -> dict:
    if not os.path.exists(CUSTOM_RUBRICS_FILE):
        return {}
    try:
        with open(CUSTOM_RUBRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def load_custom_rubrics() -> dict:
    """Returns dict of {key: rubric_dict}. Returns {} if file doesn't exist."""
    raw = get_raw_custom_rubrics()
    converted = {}
    for k, v in raw.items():
        converted[k] = {
            "name": v.get("name", k),
            "intent": v.get("description", ""),
            "parameters": [
                (p.get("name", ""), p.get("max_points", 0), p.get("description", ""))
                for p in v.get("parameters", [])
            ],
            "red_flags": [
                (rf.get("name", rf.get("description", "")), rf.get("deduction", 0))
                for rf in v.get("red_flags", [])
            ]
        }
    return converted

def save_custom_rubric(key: str, rubric: dict) -> None:
    """Adds or updates a rubric, persists to disk."""
    raw = get_raw_custom_rubrics()
    raw[key] = rubric
    with open(CUSTOM_RUBRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

def delete_custom_rubric(key: str) -> None:
    """Removes a rubric by key, persists to disk."""
    raw = get_raw_custom_rubrics()
    if key in raw:
        del raw[key]
        with open(CUSTOM_RUBRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)

def derive_key(name: str) -> str:
    """Converts 'Black Friday Campaign' → 'black_friday_campaign'."""
    name = name.lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    return name.strip('_')
