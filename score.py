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

# Objective-only deduction Red Flags (kept for backward compatibility, not used in math)
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
    Unified scoring function for all rubrics using temperature=0 and JSON response mode.
    Sets Gemini temperature=0, top_p=0.0, top_k=1, seed=42, and response_mime_type=application/json.
    Integrates persistent score caching.
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

    # 2. Build prompt parameters and instructions
    param_bullets = []
    for p in rubric_dict["parameters"]:
        param_bullets.append(
            f"- {p['key']} ({p['max_points']} pts max):\n"
            f"  Check: {p['check']}\n"
            f"  Failure modes: {', '.join(p.get('failure_modes', []))}"
        )
    param_str = "\n".join(param_bullets)
    
    prompt = f"""You are a strict, objective call-quality auditor for The Sleep Company (TSC), a premium mattress brand in India.
You audit calls to evaluate agents or bots. The calls are typically in Hindi, English, or other Indian languages (like Hinglish) — score based on meaning, not language.

CRITICAL AUDITOR INSTRUCTIONS:
- Score each parameter with an integer from 0 up to its maximum points, or "NA"
- Full points means the agent clearly and completely performed this behavior
- Partial points should be awarded if the agent partially performed the behavior but missed some aspects
- 0 points means the agent completely failed or missed this behavior
- NA means this parameter genuinely did not apply to this call (i.e. no opportunity existed for the agent to demonstrate this)
- When evidence is ambiguous, be conservative with points
- For complaint, score NA unless the customer explicitly raised a complaint during the call
- For ownership_resolution, score based only on whether the information spoken in the call was accurate and complete
- For parameters that were not discussed or reached during the call (e.g., if the call ended early, the customer hung up abruptly, or the conversation did not progress to a specific stage), score them as NA. This ensures the agent is only graded on the portion of the call that actually transpired, giving partial marks up to the point of discussion and scaling the score proportionally.
- If the call is extremely short, silent, has no actual conversation, consists only of automated messages/voicemails, or the customer hangs up immediately without any actual dialogue (e.g. immediate disconnect or wrong number), score every single parameter as NA so that the entire call is graded as NA.
- Be strict — full points should be genuinely earned.

Here is the rubric for this call:
LEAD SOURCE: {rubric_dict['name']}
CUSTOMER INTENT: {rubric_dict['description']}

SCORING PARAMETERS:
{param_str}

---
CALL TRANSCRIPT (single-speaker raw output; infer speaker turns):
{transcript_text}
---

Generate a naturally translated English version of the transcript, formatted as a dialogue where each turn starts with "Agent:" or "Customer:" on a new line with blank lines between speaker turns. "Agent" is the TSC sales representative or bot; "Customer" is the lead. Translate naturally (not word-for-word); preserve names, prices, store addresses, pincodes, and product names verbatim. If the call is already in English, clean it up (remove filler) and format as dialogue.

Force a response with a JSON object of this exact shape:
{{
  "english_transcript": "translated and formatted dialogue",
  "scores": {{
    {", ".join(f'"{p["key"]}": <integer between 0 and {p["max_points"]}, or "NA">' for p in rubric_dict["parameters"])}
  }},
  "reasons": {{
    {", ".join(f'"{p["key"]}": "1-2 sentence reason citing what was observed"' for p in rubric_dict["parameters"])}
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
                
                # Check if all keys exist and have valid values
                for p in rubric_dict["parameters"]:
                    k = p["key"]
                    if k not in result["scores"] or k not in result["reasons"]:
                        raise ValueError(f"Missing parameter key: {k}")
                    val = result["scores"][k]
                    if val != "NA":
                        try:
                            val_int = int(val)
                            if val_int < 0 or val_int > p["max_points"]:
                                raise ValueError(f"Score {val_int} out of bounds for {k}")
                            result["scores"][k] = val_int
                        except ValueError:
                            raise ValueError(f"Invalid verdict for key {k}: {val}")
                        
                scores = result["scores"]
                reasons = result["reasons"]
                summary = result["summary"]
                coaching = result["coaching"]
                english_transcript = result.get("english_transcript", "")
                
                # Compute points & final score based on new scoring math!
                yes_points = sum(scores[p["key"]] for p in rubric_dict["parameters"] if scores.get(p["key"]) != "NA")
                applicable_points = sum(p["max_points"] for p in rubric_dict["parameters"] if scores.get(p["key"]) != "NA")
                
                if applicable_points == 0:
                    final_score = "NA"
                    pass_fail = "NA"
                else:
                    final_score = round((yes_points / applicable_points) * 100)
                    pass_fail = "Pass" if final_score >= 85 else "Fail"
                
                fatal_failed = False
                fatal_failed_params = []
                
                # Fatal failure only triggers if advisor_behaviour == 0 OR ownership_resolution == 0
                for fatal_key in ["advisor_behaviour", "ownership_resolution"]:
                    if scores.get(fatal_key) == 0:  # Int 0, not "NA" or partial
                        fatal_failed = True
                        fatal_failed_params.append(fatal_key)
                
                lsq_caveat = "Note: LSQ documentation and CRM tagging cannot be verified from transcript alone. Ownership score reflects verbal accuracy in the call only."
                
                success_result = {
                    "total_score": final_score,
                    "yes_points": yes_points,
                    "applicable_points": applicable_points,
                    "fatal_failed": fatal_failed,
                    "fatal_failed_params": fatal_failed_params,
                    "pass_fail": pass_fail,
                    "summary": summary,
                    "coaching": coaching,
                    "coaching_notes": coaching,  # backward compatibility
                    "lsq_caveat": lsq_caveat,
                    "english_transcript": english_transcript,
                    "parameter_scores": {
                        p["key"]: {
                            "name": p["name"],
                            "verdict": scores.get(p["key"], "NA"),
                            "points_earned": scores.get(p["key"]) if scores.get(p["key"]) != "NA" else 0,
                            "points_max": p["max_points"],
                            "reason": reasons.get(p["key"], "")
                        } for p in rubric_dict["parameters"]
                    },
                    # backward compatibility keys:
                    "base_score": final_score,
                    "raw_score": yes_points,
                    "applicable_max": applicable_points,
                    "red_flags_triggered": [],
                    "red_flag_deduction": 0,
                    "red_flags": {},
                    "red_flag_reasons": {}
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
    lsq_caveat = "Note: LSQ documentation and CRM tagging cannot be verified from transcript alone. Ownership score reflects verbal accuracy in the call only."
    error_result = {
        "scoring_error": True,
        "raw_response": last_raw_response,
        "total_score": 0,
        "yes_points": 0,
        "applicable_points": 100,
        "fatal_failed": False,
        "fatal_failed_params": [],
        "pass_fail": "Fail",
        "lsq_caveat": lsq_caveat,
        "parameter_scores": {},
        "summary": f"SCORING ERROR: JSON validation failed. Raw response: {last_raw_response[:200]}",
        "coaching": f"Validation failed after 3 attempts on all models. Last error: {last_error}",
        "coaching_notes": f"Validation failed after 3 attempts on all models. Last error: {last_error}",
        "english_transcript": f"Error scoring call. Raw model output:\n{last_raw_response}",
        "base_score": 0,
        "raw_score": 0,
        "applicable_max": 100,
        "red_flags_triggered": [],
        "red_flag_deduction": 0,
        "red_flags": {},
        "red_flag_reasons": {}
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