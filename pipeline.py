import os
import csv
import json
import datetime
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq
from google import genai
from rubrics import format_rubric_for_prompt, RUBRICS
from score import score_transcript

def get_all_parameter_keys():
    keys = set()
    for r in RUBRICS.values():
        for p in r["parameters"]:
            keys.add(p["key"])
    return sorted(list(keys))

def transcribe_with_cache(groq_client, audio_path, filename, transcripts_dir):
    import deterministic_cache as dc
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        
    h = dc.audio_hash(audio_bytes)
    cached = dc.cache_get("transcripts", h)
    
    if cached:
        transcript_text = cached["transcript"]
        detected_language = cached["language"]
        english_translation = cached["english"]
        
        # Sync with legacy local files for backward compatibility/previews
        stem = audio_path.stem
        txt_path = transcripts_dir / f"{stem}.txt"
        lang_path = transcripts_dir / f"{stem}.lang"
        en_path = transcripts_dir / f"{stem}.en.txt"
        transcripts_dir.mkdir(exist_ok=True)
        if not txt_path.exists():
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
        if not lang_path.exists():
            with open(lang_path, "w", encoding="utf-8") as f:
                f.write(detected_language)
        if not en_path.exists():
            with open(en_path, "w", encoding="utf-8") as f:
                f.write(english_translation)
                
        return transcript_text, detected_language, english_translation

    # Cache miss
    with open(audio_path, "rb") as audio_file:
        transcript = groq_client.audio.transcriptions.create(
            file=(filename, audio_file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            temperature=0.0,
        )
    transcript_text = transcript.text
    detected_language = getattr(transcript, "language", "")
    
    with open(audio_path, "rb") as audio_file:
        try:
            translation = groq_client.audio.translations.create(
                file=(filename, audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
                temperature=0.0,
            )
            english_translation = translation.text
        except Exception:
            english_translation = transcript_text
            
    # Save to cache
    dc.cache_set("transcripts", h, {
        "transcript": transcript_text,
        "language": detected_language,
        "english": english_translation
    })
    
    # Save to legacy local files
    stem = audio_path.stem
    txt_path = transcripts_dir / f"{stem}.txt"
    lang_path = transcripts_dir / f"{stem}.lang"
    en_path = transcripts_dir / f"{stem}.en.txt"
    transcripts_dir.mkdir(exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    with open(lang_path, "w", encoding="utf-8") as f:
        f.write(detected_language)
    with open(en_path, "w", encoding="utf-8") as f:
        f.write(english_translation)
        
    return transcript_text, detected_language, english_translation

def main():
    load_dotenv()
    
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    recordings_csv = Path("recordings.csv")
    recordings_dir = Path("recordings")
    transcripts_dir = Path("transcripts")
    output_dir = Path("output")
    
    transcripts_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    today_str = datetime.date.today().isoformat()
    report_csv_path = output_dir / f"audit-report-{today_str}.csv"
    error_log_path = output_dir / f"errors-{today_str}.log"
    
    valid_lead_sources = set(RUBRICS.keys())
    
    if not recordings_csv.exists():
        print(f"{recordings_csv} not found.")
        return
        
    with open(recordings_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    all_params = get_all_parameter_keys()
    csv_headers = [
        "filename",
        "lead_source",
        "agent_id",
        "total_score",
        "fatal_failed",
        "fatal_failed_params",
        "red_flags_triggered",
        "red_flag_deduction"
    ] + all_params + [
        "summary",
        "coaching"
    ]
    
    write_header = not report_csv_path.exists()
    
    out_f = open(report_csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=csv_headers, extrasaction="ignore")
    if write_header:
        writer.writeheader()
        
    total_processed = 0
    succeeded = 0
    failed = 0
    
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
            
            rubric_dict = RUBRICS[lead_source]
            result, used_model = score_transcript(transcript_text, rubric_dict, gemini_client)
            
            if result.get("scoring_error"):
                raise RuntimeError(result.get("coaching_notes", "Unknown scoring error"))
                
            out_row = {
                "filename": filename,
                "lead_source": lead_source,
                "agent_id": agent_id,
                "total_score": result.get("total_score", 0),
                "fatal_failed": str(result.get("fatal_failed", False)),
                "fatal_failed_params": ", ".join(result.get("fatal_failed_params", [])),
                "red_flags_triggered": ", ".join(result.get("red_flags_triggered", [])),
                "red_flag_deduction": result.get("red_flag_deduction", 0),
                "summary": result.get("summary", ""),
                "coaching": result.get("coaching", "")
            }
            
            ps = result.get("parameter_scores", {})
            for param_key in all_params:
                if param_key in ps:
                    out_row[param_key] = ps[param_key]["verdict"]
                else:
                    out_row[param_key] = ""
                    
            writer.writerow(out_row)
            out_f.flush()
            succeeded += 1
            
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
