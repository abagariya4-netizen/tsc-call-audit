import os
import csv
import json
import datetime
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq
from google import genai
from google.genai import types
from rubrics import format_rubric_for_prompt, RUBRICS

def get_all_parameter_names():
    names = set()
    for r in RUBRICS.values():
        for p in r["parameters"]:
            names.add(p[0])
    return sorted(list(names))

def transcribe_with_cache(groq_client, audio_path, filename, transcripts_dir):
    stem = audio_path.stem
    txt_path = transcripts_dir / f"{stem}.txt"
    lang_path = transcripts_dir / f"{stem}.lang"
    en_path = transcripts_dir / f"{stem}.en.txt"
    
    if txt_path.exists() and lang_path.exists():
        with open(txt_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()
        with open(lang_path, "r", encoding="utf-8") as f:
            detected_language = f.read().strip()
    else:
        with open(audio_path, "rb") as audio_file:
            transcript = groq_client.audio.transcriptions.create(
                file=(filename, audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
        transcript_text = transcript.text
        detected_language = getattr(transcript, "language", "")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        with open(lang_path, "w", encoding="utf-8") as f:
            f.write(detected_language)
            
    if en_path.exists():
        with open(en_path, "r", encoding="utf-8") as f:
            english_translation = f.read()
    else:
        with open(audio_path, "rb") as audio_file:
            translation = groq_client.audio.translations.create(
                file=(filename, audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
        english_translation = translation.text
        with open(en_path, "w", encoding="utf-8") as f:
            f.write(english_translation)
            
    return transcript_text, detected_language, english_translation

def main():
    load_dotenv()
    
    # Configure Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # Configure Gemini using new SDK
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    recordings_csv = Path("recordings.csv")
    recordings_dir = Path("recordings")
    transcripts_dir = Path("transcripts")
    output_dir = Path("output")
    
    transcripts_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    today_str = datetime.date.today().isoformat()
    report_csv_path = output_dir / f"audit-report-{today_str}.csv"
    error_log_path = output_dir / f"errors-{today_str}.log"
    
    valid_lead_sources = {
        "find_a_store", "arrange_callback", "inbound", 
        "shopflo_abandoned_cart", "next_day_delivery", "no_cost_emi"
    }
    
    if not recordings_csv.exists():
        print(f"{recordings_csv} not found.")
        return
        
    with open(recordings_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    all_params = get_all_parameter_names()
    csv_headers = [
        "filename", "agent_id", "lead_source", "detected_language", "total_score"
    ] + all_params + ["red_flags_triggered", "summary", "coaching_notes", "model_used"]
    
    write_header = not report_csv_path.exists()
    
    out_f = open(report_csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=csv_headers)
    if write_header:
        writer.writeheader()
        
    total_processed = 0
    succeeded = 0
    failed = 0
    
    models_to_try = [
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-3-flash-preview"
    ]
    
    for row in tqdm(rows, desc="Processing calls"):
        filename = row.get("filename", "").strip()
        lead_source = row.get("lead_source", "").strip()
        agent_id = row.get("agent_id", "").strip()
        
        audio_path = recordings_dir / filename
        
        if not audio_path.exists():
            print(f"Warning: Missing {audio_path}, skipping.")
            continue
            
        if not lead_source or lead_source not in valid_lead_sources:
            print(f"Warning: Invalid or empty lead_source '{lead_source}' for {filename}, skipping.")
            continue
            
        total_processed += 1
        
        try:
            transcript_text, detected_language, english_translation = transcribe_with_cache(
                groq_client, audio_path, filename, transcripts_dir
            )
            
            rubric_text = format_rubric_for_prompt(lead_source)
            
            prompt = f"""You are a strict call-quality auditor for The Sleep Company (TSC),
a premium mattress brand in India. You audit calls made by outbound and inbound
sales agents to leads. The calls are typically in Hindi, English, or other
Indian languages — score based on meaning, not language.

Here is the rubric for the lead source of this call:

{rubric_text}

Here is the call transcript (single-speaker raw output; you must infer
which lines are agent vs customer):

---
{transcript_text}
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
            
            result = None
            used_model = None
            last_raw_response = ""
            
            for m_name in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=m_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                        ),
                    )
                    last_raw_response = response.text
                    result = json.loads(response.text)
                    used_model = m_name
                    break
                except json.JSONDecodeError:
                    raise
                except Exception as e:
                    # Generic catch for API exceptions to fallback to the next model
                    continue
                    
            if result is None:
                raise RuntimeError("All models failed due to ResourceExhausted or NotFound.")
                
            out_row = {
                "filename": filename,
                "agent_id": agent_id,
                "lead_source": lead_source,
                "detected_language": detected_language,
                "total_score": result.get("total_score", 0),
                "summary": result.get("summary", ""),
                "coaching_notes": result.get("coaching_notes", ""),
                "model_used": used_model
            }
            
            for param in result.get("parameters", []):
                p_name = param.get("name")
                if p_name in all_params:
                    out_row[p_name] = f"{param.get('score', 0)}/{param.get('max', 0)} - {param.get('reason', '')}"
                    
            red_flags = []
            for rf in result.get("red_flags_triggered", []):
                red_flags.append(f"{rf.get('description')} ({rf.get('penalty')})")
            out_row["red_flags_triggered"] = "; ".join(red_flags)
            
            writer.writerow(out_row)
            out_f.flush()
            succeeded += 1
            
        except json.JSONDecodeError as e:
            err_msg = f"JSON parse failed. Raw response: {last_raw_response}"
            with open(error_log_path, "a", encoding="utf-8") as ef:
                ef.write(f"[{datetime.datetime.now().isoformat()}] {filename} - {e.__class__.__name__} - {err_msg}\n")
            failed += 1
        except Exception as e:
            with open(error_log_path, "a", encoding="utf-8") as ef:
                ef.write(f"[{datetime.datetime.now().isoformat()}] {filename} - {e.__class__.__name__} - {str(e)}\n")
            failed += 1
            
    out_f.close()
    
    print("\n--- Pipeline Complete ---")
    print(f"Total Processed: {total_processed}")
    print(f"Succeeded:       {succeeded}")
    print(f"Failed:          {failed}")
    print(f"Output CSV Path: {report_csv_path}")

if __name__ == "__main__":
    main()
