import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from rubrics import get_rubric, RUBRICS, RUBRIC_VERSION
import deterministic_cache as dc

# Load secrets
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_FALLBACK_LADDER = [
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
]

# Objective-only deduction Red Flags
RED_FLAGS = [
    {
        "key": "rude_to_customer",
        "name": "Rude/abusive to customer",
        "deduction": -10,
        "check": "Did the agent speak rudely, dismissively, or abusively to the customer (insults, sarcasm, hostility, talking down)?"
    },
    {
        "key": "wrong_info",
        "name": "Gave wrong price/product information",
        "deduction": -10,
        "check": "Did the agent state clearly incorrect price or product information (wrong price, non-existent product, false specification)?"
    }
]

def score_transcript(transcript_text, rubric_dict, gemini_client, english_transcript=None):
    """
    Unified scoring function for all 7 rubrics using temperature=0 and JSON response mode.
    Sets Gemini temperature=0, top_p=0.0, top_k=1, seed=42, and response_mime_type=application/json.
    Integrates persistent score caching and deduction-only red flags.
    """
    # 1. Cache look-up
    rubric_key = None
    for k, v in RUBRICS.items():
        if v == rubric_dict:
            rubric_key = k
            break
    if not rubric_key:
        rubric_key = rubric_dict.get("name", "unknown").lower().replace(" ", "_")
        
    text_to_hash = english_transcript if english_transcript else transcript_text
    cache_key = dc.text_hash(text_to_hash, rubric_key, RUBRIC_VERSION)
    
    cached_score = dc.cache_get("scores", cache_key)
    if cached_score:
        return cached_score["result"], cached_score["model_used"]

    # 2. Build prompt parameters and red flags descriptions
    param_bullets = []
    for p in rubric_dict["parameters"]:
        param_bullets.append(
            f"- {p['key']} ({p['max_points']} pts max):\n"
            f"  Check: {p['check']}\n"
            f"  Failure modes: {', '.join(p.get('failure_modes', []))}"
        )
    param_str = "\n".join(param_bullets)
    
    rf_bullets = []
    for rf in RED_FLAGS:
        rf_bullets.append(
            f"- {rf['key']} ({rf['deduction']} pts penalty):\n"
            f"  Check: {rf['check']}"
        )
    rf_str = "\n".join(rf_bullets)
    
    prompt = f"""You are a strict, objective call-quality auditor for The Sleep Company (TSC), a premium mattress brand in India.
You audit calls to evaluate agents or bots. The calls are typically in Hindi, English, or other Indian languages (like Hinglish) — score based on meaning, not language.

CRITICAL AUDITOR INSTRUCTIONS:
1. Score "Yes", "No", or "NA" ONLY for scoring parameters. Do not use any other values.
2. Be extremely strict and objective: a 100/100 should be rare and represent an absolutely flawless call.
3. Default to "No" when the evidence for a parameter is ambiguous, weak, or incomplete. Do NOT give the benefit of the doubt.
4. For the "complaint" parameter: Score "NA" unless the customer explicitly raised a complaint or expressed frustration. Do not score "Yes" or "No" unless an active complaint or grievance was present in the call.
5. For the "ownership_resolution" parameter: Evaluate ONLY the verbal accuracy and completeness of the information provided by the agent during the call itself. Do NOT assume any post-call CRM action was completed or not completed, as that cannot be verified from the audio transcript alone.
6. Translate naturally and accurately: preserve names, prices, store addresses, pincodes, and product names verbatim in the English translation.
7. For red flags, answer true ONLY if there is explicit evidence in the transcript. Otherwise, default to false.

Here is the rubric for this call:
LEAD SOURCE: {rubric_dict['name']}
CUSTOMER INTENT: {rubric_dict['description']}

SCORING PARAMETERS:
{param_str}

RED FLAGS (answer true/false if triggered):
{rf_str}

---
CALL TRANSCRIPT (single-speaker raw output; infer speaker turns):
{transcript_text}
---

Generate a naturally translated English version of the transcript, formatted as a dialogue where each turn starts with "Agent:" or "Customer:" on a new line with blank lines between speaker turns. "Agent" is the TSC sales representative or bot; "Customer" is the lead. Translate naturally (not word-for-word); preserve names, prices, store addresses, pincodes, and product names verbatim. If the call is already in English, clean it up (remove filler) and format as dialogue.

SCORING RULES (CRITICAL FOR DETERMINISM):
1. For each parameter, evaluate the check criteria strictly.
2. Answer "Yes" ONLY if there is clear, explicit evidence in the transcript that the agent met the criteria.
3. Answer "No" if there is evidence of failure or if the required action was missing.
4. Answer "NA" ONLY if the parameter is genuinely not applicable to the call.
5. Base your verdict strictly on what is explicitly present in the transcript. Do not infer, assume, or give the benefit of the doubt.
6. When evidence is ambiguous or missing, default to "No" (deterministic default).
7. For red flags, answer true ONLY if there is explicit evidence in the transcript. Otherwise, default to false.

Force a response with a JSON object of this exact shape:
{{
  "english_transcript": "translated and formatted dialogue",
  "scores": {{
    {", ".join(f'"{p["key"]}": "Yes" | "No" | "NA"' for p in rubric_dict["parameters"])}
  }},
  "reasons": {{
    {", ".join(f'"{p["key"]}": "1-2 sentence reason citing what was observed"' for p in rubric_dict["parameters"])}
  }},
  "red_flags": {{
    "rude_to_customer": true | false,
    "wrong_info": true | false
  }},
  "red_flag_reasons": {{
    "rude_to_customer": "evidence or empty string",
    "wrong_info": "evidence or empty string"
  }},
  "summary": "2-3 sentence overall call summary",
  "coaching": "2-3 specific coaching points for the agent"
}}
"""
    last_error = None
    last_raw_response = ""
    for model_name in MODEL_FALLBACK_LADDER:
        for attempt in range(3):
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        top_p=0.0,
                        top_k=1,
                        response_mime_type="application/json",
                        seed=42
                    ),
                )
                last_raw_response = response.text.strip()
                if last_raw_response.startswith("```"):
                    last_raw_response = last_raw_response.strip("`")
                    if last_raw_response.startswith("json"):
                        last_raw_response = last_raw_response[4:].strip()
                        
                result = json.loads(last_raw_response)
                
                # Schema validation
                if "scores" not in result or "reasons" not in result or "summary" not in result or "coaching" not in result:
                    raise ValueError("JSON response missing top-level keys")
                if "red_flags" not in result or "red_flag_reasons" not in result:
                    raise ValueError("JSON response missing red flag keys")
                
                # Check if all keys exist and have valid values
                for p in rubric_dict["parameters"]:
                    k = p["key"]
                    if k not in result["scores"] or k not in result["reasons"]:
                        raise ValueError(f"Missing parameter key: {k}")
                    if result["scores"][k] not in ("Yes", "No", "NA"):
                        raise ValueError(f"Invalid verdict for key {k}: {result['scores'][k]}")
                        
                # Check and normalize red flags
                for rf in RED_FLAGS:
                    rk = rf["key"]
                    if rk not in result["red_flags"] or rk not in result["red_flag_reasons"]:
                        raise ValueError(f"Missing red flag key: {rk}")
                    if not isinstance(result["red_flags"][rk], bool):
                        val_str = str(result["red_flags"][rk]).strip().lower()
                        result["red_flags"][rk] = val_str in ("true", "1", "yes")
                
                scores = result["scores"]
                reasons = result["reasons"]
                red_flags_data = result["red_flags"]
                red_flag_reasons = result["red_flag_reasons"]
                summary = result["summary"]
                coaching = result["coaching"]
                english_transcript = result.get("english_transcript", "")
                
                # Compute base score in Python (NOT in Gemini)
                raw_total = sum(p["max_points"] for p in rubric_dict["parameters"]
                               if scores.get(p["key"]) == "Yes")
                applicable_max = sum(p["max_points"] for p in rubric_dict["parameters"]
                                    if scores.get(p["key"]) != "NA")
                
                base_score = round((raw_total / applicable_max) * 100) if applicable_max > 0 else 0
                
                # Compute red flag deduction
                red_flags_triggered = [rf["key"] for rf in RED_FLAGS if red_flags_data.get(rf["key"]) is True]
                red_flag_deduction = sum(rf["deduction"] for rf in RED_FLAGS if red_flags_data.get(rf["key"]) is True)
                
                # Check for fatal failures
                fatal_failed = False
                fatal_failed_params = []
                for p in rubric_dict["parameters"]:
                    if p.get("fatal"):
                        verdict = str(scores.get(p["key"], "NA")).strip()
                        if verdict.lower() == "no":
                            fatal_failed = True
                            fatal_failed_params.append(p["name"])
                
                if fatal_failed:
                    final_score = 0
                else:
                    final_score = max(0, base_score + red_flag_deduction)
                
                lsq_caveat = "Note: CRM tagging, LeadSquared logging, and actual hold/mute time cannot be fully verified from call audio alone. These parameters require secondary verification in LeadSquared CRM."
                
                success_result = {
                    "base_score": base_score,
                    "total_score": final_score,
                    "raw_score": raw_total,
                    "applicable_max": applicable_max,
                    "red_flags_triggered": red_flags_triggered,
                    "red_flag_deduction": red_flag_deduction,
                    "red_flags": red_flags_data,
                    "red_flag_reasons": red_flag_reasons,
                    "fatal_failed": fatal_failed,
                    "fatal_failed_params": fatal_failed_params,
                    "lsq_caveat": lsq_caveat,
                    "parameter_scores": {
                        p["key"]: {
                            "name": p["name"],
                            "verdict": scores.get(p["key"], "NA"),
                            "points_earned": p["max_points"] if scores.get(p["key"]) == "Yes" else 0,
                            "points_max": p["max_points"],
                            "reason": reasons.get(p["key"], "")
                        } for p in rubric_dict["parameters"]
                    },
                    "summary": summary,
                    "coaching_notes": coaching,
                    "coaching": coaching,
                    "english_transcript": english_transcript,
                    "pass_fail": "Pass" if final_score >= 85 else "Fail"
                }
                
                # Save to cache
                dc.cache_set("scores", cache_key, {
                    "result": success_result,
                    "model_used": model_name
                })
                
                return success_result, model_name
            except Exception as e:
                last_error = e
                continue
                
    # All retries failed
    error_result = {
        "scoring_error": True,
        "raw_response": last_raw_response,
        "base_score": 0,
        "total_score": 0,
        "pass_fail": "Fail",
        "raw_score": 0,
        "applicable_max": 100,
        "fatal_failed": False,
        "fatal_failed_params": [],
        "red_flags_triggered": [],
        "red_flag_deduction": 0,
        "red_flags": {rf["key"]: False for rf in RED_FLAGS},
        "red_flag_reasons": {rf["key"]: "" for rf in RED_FLAGS},
        "lsq_caveat": "Note: CRM tagging, LeadSquared logging, and actual hold/mute time cannot be fully verified from call audio alone. These parameters require secondary verification in LeadSquared CRM.",
        "parameter_scores": {},
        "summary": f"SCORING ERROR: JSON validation failed. Raw response: {last_raw_response[:200]}",
        "coaching_notes": f"Validation failed after 3 attempts on all models. Last error: {last_error}",
        "coaching": f"Validation failed after 3 attempts on all models. Last error: {last_error}",
        "english_transcript": f"Error scoring call. Raw model output:\n{last_raw_response}"
    }
    return error_result, "gemini-3.1-flash-lite"

# Standalone execution
if __name__ == "__main__":
    LEAD_SOURCE = "find_a_store"
    TRANSCRIPT = """
    Agent: Good afternoon, thank you for calling The Sleep Company. My name is Rohan. Am I speaking with Mr. Verma?
    Customer: Yes, Rohan. I was looking at your mattresses online.
    Agent: Perfect. Understood, sir. Are you looking for a king size or queen size mattress?
    Customer: King size. But my budget is around 20,000.
    Agent: I see. Sir, our SmartGrid Ortho is excellent for back pain, and it comes perfectly in your range with our current bank offers. We also offer 0% interest No-Cost EMI options.
    Customer: Do you have delivery charges?
    Agent: Absolutely not. Delivery and installation are completely free, plus you get a 100-night risk-free trial.
    Customer: That sounds good. Can you send me the details on WhatsApp?
    Agent: Yes, I will share the nearest store address with you where you can experience the mattress, and we can note down details. What is your preferred date to visit the store, Mr. Verma?
    Customer: I can visit tomorrow.
    Agent: Perfect, I have locked in your visit for tomorrow afternoon at our Sector 15 store. I will send the address immediately on this number. Sir, is there any other help you need?
    Customer: No, that's it.
    Agent: Excellent, Mr. Verma. Have a great day ahead!
    """
    
    print(f"Scoring transcript against rubric: {LEAD_SOURCE}")
    print("Asking Gemini... this takes 5-15 seconds.\n")
    
    rubric = get_rubric(LEAD_SOURCE)
    result, model_used = score_transcript(TRANSCRIPT, rubric, client)
    
    print("=" * 60)
    print(f"LEAD SOURCE: {LEAD_SOURCE}")
    print(f"TOTAL SCORE: {result['total_score']} / 100")
    print(f"FATAL FAILED: {result.get('fatal_failed', False)}")
    print("=" * 60)
    
    print("\nPARAMETER SCORES:")
    for k, p in result["parameter_scores"].items():
        print(f"  {k} ({p['name']}): {p['verdict']} (points: {p['points_earned']}/{p['points_max']}) — {p['reason']}")
            
    print("\nSUMMARY:")
    print(f"  {result['summary']}")
    print("\nCOACHING NOTES:")
    print(f"  {result['coaching_notes']}")
    print("=" * 60)