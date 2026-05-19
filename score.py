import os
import json
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
from rubrics import format_rubric_for_prompt

# Load secrets
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Which lead source to score against (we'll make this dynamic later)
LEAD_SOURCE = "find_a_store"

# For this first test, paste a transcript directly here.
# Later, score.py will read transcripts from disk.
TRANSCRIPT = """
PASTE YOUR TRANSCRIPT HERE
"""

# Build the prompt
rubric_text = format_rubric_for_prompt(LEAD_SOURCE)

prompt = f"""You are a strict call-quality auditor for The Sleep Company (TSC),
a premium mattress brand in India. You audit calls made by outbound and inbound
sales agents to leads. The calls are typically in Hindi, English, or other
Indian languages — score based on meaning, not language.

Here is the rubric for the lead source of this call:

{rubric_text}

Here is the call transcript (single-speaker raw output; you must infer
which lines are agent vs customer):

---
{TRANSCRIPT}
---

Score this call against the rubric. For each parameter, give:
- "score": integer points awarded (0 to the max for that parameter)
- "reason": one short sentence explaining the score

Also list any red flags that apply (with their deduction amount).

Finally compute the total score (sum of parameters MINUS red flag deductions,
floored at 0).

Return ONLY valid JSON in this exact structure, with no markdown fences,
no extra commentary:

{{
  "parameters": [
    {{"name": "...", "score": 0, "max": 0, "reason": "..."}}
  ],
  "red_flags_triggered": [
    {{"description": "...", "penalty": 0}}
  ],
  "total_score": 0,
  "summary": "2-3 sentence summary of agent's performance",
  "coaching_notes": "2-3 specific things the agent should improve"
}}
"""

print(f"Scoring transcript against rubric: {LEAD_SOURCE}")
print("Asking Gemini... this takes 5-15 seconds.\n")

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
    ),
)

# Gemini sometimes wraps JSON in ```json ... ``` — strip that if so
raw_text = response.text.strip()
if raw_text.startswith("```"):
    raw_text = raw_text.strip("`")
    if raw_text.startswith("json"):
        raw_text = raw_text[4:].strip()

try:
    # Parse the JSON Gemini returned
    result = json.loads(raw_text)
except json.JSONDecodeError as e:
    print(f"Failed to parse JSON. Raw output from model:\n{response.text}")
    raise e

# Pretty-print the result
print("=" * 60)
print(f"LEAD SOURCE: {LEAD_SOURCE}")
print(f"TOTAL SCORE: {result['total_score']} / 100")
print("=" * 60)
print("\nPARAMETER SCORES:")
for p in result["parameters"]:
    print(f"  {p['name']}: {p['score']}/{p['max']}  — {p['reason']}")

if result["red_flags_triggered"]:
    print("\nRED FLAGS:")
    for rf in result["red_flags_triggered"]:
        print(f"  ({rf['penalty']}) {rf['description']}")
else:
    print("\nRED FLAGS: none")

print("\nSUMMARY:")
print(f"  {result['summary']}")
print("\nCOACHING NOTES:")
print(f"  {result['coaching_notes']}")
print("=" * 60)