"""
TSC Call Audit Dashboard

A Streamlit dashboard that:
1. Drag-and-drop one or many call recordings
2. Auto-detects the lead source per file (with manual override per file)
3. Processes files in parallel (up to 4 at a time)
4. Shows the results inline and appends to today's audit-report CSV
"""

import os
import csv
import json
import time
import io
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from google import genai
from google.genai import types

from rubrics import format_rubric_for_prompt, RUBRICS
from custom_rubrics_storage import load_custom_rubrics, save_custom_rubric, delete_custom_rubric, derive_key
from score import score_transcript

# Load and convert custom rubrics to the new dict structure
def load_and_convert_custom_rubrics():
    custom = load_custom_rubrics()
    converted = {}
    for k, v in custom.items():
        params = []
        for p in v["parameters"]:
            if isinstance(p, tuple):
                name, max_points, check = p
                key = derive_key(name)
                params.append({
                    "key": key,
                    "name": name,
                    "max_points": max_points,
                    "fatal": False,
                    "check": check,
                    "failure_modes": []
                })
            else:
                params.append(p)
        converted[k] = {
            "name": v["name"],
            "description": v.get("intent", ""),
            "parameters": params
        }
    return converted

CUSTOM_RUBRICS = load_and_convert_custom_rubrics()
# Clean up deleted rubrics from the cached module dictionary
BUILT_IN_KEYS = {"find_a_store", "arrange_callback", "inbound", "shopflo_abandoned_cart", "next_day_delivery", "no_cost_emi", "ai_voice_bot"}
for k in list(RUBRICS.keys()):
    if k not in BUILT_IN_KEYS and k not in CUSTOM_RUBRICS:
        del RUBRICS[k]
RUBRICS.update(CUSTOM_RUBRICS)

# ============================================================
# SETUP
# ============================================================

load_dotenv()

@st.cache_resource
def get_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@st.cache_resource
def get_csv_lock():
    return threading.Lock()

groq_client = get_groq_client()
gemini_client = get_gemini_client()
csv_lock = get_csv_lock()

MODEL_FALLBACK_LADDER = [
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
]

# "temperature": 0

LEAD_SOURCE_KEYS = list(RUBRICS.keys())

AUTO_DETECT_KEY = "__auto__"
AI_CATEGORY_KEYS = {"ai_voice_bot"}

UPLOAD_CHOICES = [AUTO_DETECT_KEY, "ai_voice_bot"]

MAX_PARALLEL = 4

def get_all_parameter_keys():
    keys = set()
    for r in RUBRICS.values():
        for p in r["parameters"]:
            keys.add(p["key"])
    return sorted(list(keys))

META_COLS = {
    "filename", "agent_id", "lead_source", "detected_language",
    "total_score", "summary", "coaching",
    "model_used", "fatal_failed", "fatal_failed_params", "raw_score",
}

# ============================================================
# HELPERS
# ============================================================

def format_lead_source_choice(k):
    if k == AUTO_DETECT_KEY:
        return "🤖 Auto-detect from audio"
    if k in AI_CATEGORY_KEYS:
        return f"🎙️ {RUBRICS[k]['name']}"
    return f"👤 {RUBRICS[k]['name']}"


def transcribe_audio(file_bytes, filename):
    import deterministic_cache as dc
    h = dc.audio_hash(file_bytes)
    cached = dc.cache_get("transcripts", h)
    if cached:
        return cached["transcript"], cached["language"], cached["english"]
        
    transcript = groq_client.audio.transcriptions.create(
        file=(filename, file_bytes),
        model="whisper-large-v3",
        response_format="verbose_json",
        temperature=0.0,
    )
    try:
        translation = groq_client.audio.translations.create(
            file=(filename, file_bytes),
            model="whisper-large-v3",
            response_format="verbose_json",
            temperature=0.0,
        )
        english_translation = translation.text
    except Exception:
        english_translation = transcript.text
        
    dc.cache_set("transcripts", h, {
        "transcript": transcript.text,
        "language": transcript.language,
        "english": english_translation
    })
    return transcript.text, transcript.language, english_translation


def auto_classify_and_score(transcript_text):
    """Classify lead source then score."""
    lead_source = classify_transcript(transcript_text)
    rubric_dict = RUBRICS[lead_source]
    result, model_used = score_transcript(transcript_text, rubric_dict, gemini_client)
    return result, model_used, lead_source


def classify_transcript(english_transcript):
    import deterministic_cache as dc
    all_keys = sorted([k for k in RUBRICS if k != "ai_voice_bot"])
    
    # 1. Classification cache check
    key_parts = [english_transcript] + all_keys
    h = dc.text_hash(*key_parts)
    cached = dc.cache_get("classifications", h)
    if cached:
        return cached["lead_source"]
        
    desc_list = []
    for k in all_keys:
        desc = RUBRICS[k].get("description", "")
        desc_list.append(f"- {k}: {desc}")
    desc_str = "\n".join(desc_list)
    keys_str = ", ".join(f"'{k}'" for k in all_keys)

    prompt = f"""You are a strict call classifier. Classify this call into exactly ONE lead source category from the allowed categories listed below. Return ONLY valid JSON matching the exact schema specified.

Allowed categories:
{keys_str}

Category descriptions:
{desc_str}

Transcript:
{english_transcript}

Respond with a JSON object of this exact shape:
{{"lead_source": "<key>"}}

CRITICAL INSTRUCTIONS:
1. The value for "lead_source" MUST be EXACTLY one of the allowed categories: {keys_str}. Do not invent new categories.
2. Respond with EXACTLY one of these categories and nothing else in the JSON.
3. If you are uncertain or if multiple categories could apply, pick the FIRST listed category that could plausibly apply (deterministic tie-breaking).
"""

    last_error = None
    for model_name in MODEL_FALLBACK_LADDER:
        for _attempt in range(3):
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
                data = json.loads(response.text.strip())
                val = str(data.get("lead_source", "")).strip().lower()
                if val in RUBRICS and val != "ai_voice_bot":
                    dc.cache_set("classifications", h, {"lead_source": val})
                    return val
                for k in all_keys:
                    if k in val:
                        dc.cache_set("classifications", h, {"lead_source": k})
                        return k
            except Exception as e:
                last_error = e
                continue
                
    fallback_val = all_keys[0] if all_keys else "find_a_store"
    dc.cache_set("classifications", h, {"lead_source": fallback_val})
    return fallback_val


def build_csv_row(filename, agent_id, lead_source, language, result, model_used):
    row = {
        "filename": filename,
        "lead_source": lead_source,
        "agent_id": agent_id,
        "total_score": result.get("total_score", 0),
        "pass_fail": result.get("pass_fail", ""),
        "fatal_failed": str(result.get("fatal_failed", False)),
        "fatal_failed_params": ", ".join(result.get("fatal_failed_params", [])),
        "red_flags_triggered": ", ".join(result.get("red_flags_triggered", [])),
        "red_flag_deduction": result.get("red_flag_deduction", 0),
        "summary": result.get("summary", ""),
        "coaching": result.get("coaching", ""),
        "detected_language": language,
        "model_used": model_used,
        "raw_score": result.get("raw_score", 0)
    }
    
    ps = result.get("parameter_scores", {})
    for p_key in get_all_parameter_keys():
        if p_key in ps:
            row[p_key] = ps[p_key]["verdict"]
        else:
            row[p_key] = ""
            
    return row


def _do_append_to_today_csv(row_dict):
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / f"audit-report-{today}.csv"
    file_exists = csv_path.exists()

    all_params = get_all_parameter_keys()
    columns = (
        ["filename", "lead_source", "agent_id", "total_score", "pass_fail", "fatal_failed", "fatal_failed_params"]
        + all_params
        + ["red_flags_triggered", "red_flag_deduction"]
        + ["summary", "coaching", "detected_language", "model_used", "raw_score"]
    )

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({c: row_dict.get(c, "") for c in columns})
    return csv_path


def append_to_today_csv(row_dict, max_retries=5, delay=1.0):
    """Append with lock (for parallel safety) and retry (for Windows file-lock when CSV open in Excel)."""
    with csv_lock:
        last_error = None
        for attempt in range(max_retries):
            try:
                return _do_append_to_today_csv(row_dict)
            except PermissionError as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(delay)
        raise PermissionError(
            f"Could not write to CSV after {max_retries} attempts — "
            "the file might be open in Excel or another program. "
            "Close it and click Score again. "
            f"Original error: {last_error}"
        ) from last_error


def save_recording_and_transcript(file_bytes, original_filename, transcript_text, language, english_transcript=None):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    safe_filename = f"upload-{timestamp}-{original_filename}"
    safe_stem = Path(safe_filename).stem
    Path("recordings").mkdir(exist_ok=True)
    with open(Path("recordings") / safe_filename, "wb") as f:
        f.write(file_bytes)
    Path("transcripts").mkdir(exist_ok=True)
    with open(Path("transcripts") / f"{safe_stem}.txt", "w", encoding="utf-8") as f:
        f.write(transcript_text)
    with open(Path("transcripts") / f"{safe_stem}.lang", "w", encoding="utf-8") as f:
        f.write(language)
    if english_transcript:
        with open(Path("transcripts") / f"{safe_stem}.en.txt", "w", encoding="utf-8") as f:
            f.write(english_transcript)
    return safe_filename


def fetch_audio_from_url(url: str, timeout: int = 120) -> tuple[bytes, str]:
    """Download audio from a URL. Returns (file_bytes, derived_filename).
    The filename is used to hint the audio format to Groq Whisper."""
    import requests
    from urllib.parse import urlparse, unquote
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    file_bytes = response.content
    path = unquote(urlparse(url).path)
    filename = path.rsplit("/", 1)[-1] or "audio.mp3"
    filename = filename.split("?")[0]
    if "." not in filename:
        filename += ".mp3"
    return file_bytes, filename


def process_one_file(uploaded_name, uploaded_bytes, lead_source_or_auto, agent_id, csv_driven_flag=False, cached_data=None):
    """Full pipeline for one file. Designed to run in a worker thread."""
    if cached_data:
        transcript_text = cached_data["orig"]
        detected_language = cached_data["lang"]
        english_transcript = cached_data["eng"]
    else:
        transcript_text, detected_language, english_transcript = transcribe_audio(uploaded_bytes, uploaded_name)

    auto_detected = False
    if lead_source_or_auto == AUTO_DETECT_KEY:
        result, model_used, lead_source = auto_classify_and_score(transcript_text)
        auto_detected = True
        english_transcript = result.get("english_transcript", english_transcript)
    else:
        lead_source = lead_source_or_auto
        rubric_dict = RUBRICS[lead_source]
        result, model_used = score_transcript(transcript_text, rubric_dict, gemini_client)
        english_transcript = result.get("english_transcript", english_transcript)

    safe_filename = save_recording_and_transcript(
        uploaded_bytes, uploaded_name, transcript_text, detected_language, english_transcript
    )
    persistent_row = build_csv_row(
        filename=safe_filename,
        agent_id=agent_id or "unknown",
        lead_source=lead_source,
        language=detected_language,
        result=result,
        model_used=model_used,
    )
    append_to_today_csv(persistent_row)
    display_row = build_csv_row(
        filename=uploaded_name,
        agent_id=agent_id or "unknown",
        lead_source=lead_source,
        language=detected_language,
        result=result,
        model_used=model_used,
    )
    return {
        "filename": uploaded_name,
        "saved_filename": safe_filename,
        "score": result.get("total_score", 0),
        "lead_source": lead_source,
        "auto_detected": auto_detected,
        "csv_driven_flag": csv_driven_flag,
        "row": display_row,
        "transcript": transcript_text,
        "english_transcript": english_transcript,
        "parameter_scores": result.get("parameter_scores")
    }


def load_transcripts_from_disk(filename):
    if not filename or not isinstance(filename, str):
        return None, None
    stem = Path(filename).stem
    txt_path = Path("transcripts") / f"{stem}.txt"
    en_path = Path("transcripts") / f"{stem}.en.txt"
    
    orig = None
    if txt_path.exists():
        try:
            orig = txt_path.read_text(encoding="utf-8")
        except Exception:
            pass
            
    en = None
    if en_path.exists():
        try:
            en = en_path.read_text(encoding="utf-8")
        except Exception:
            pass
            
    return orig, en


def render_call_card(row, transcript_text=None, english_transcript=None, parameter_scores=None):
    score = int(row["total_score"])
    pass_fail = row.get("pass_fail", "")
    
    if score >= 80:
        score_emoji = "🟢"
    elif score >= 50:
        score_emoji = "🟡"
    else:
        score_emoji = "🔴"
    
    badge_html = ""
    if pass_fail == "Pass":
        badge_html = "<span style='color: green; font-weight: bold;'>PASS</span>"
    elif pass_fail == "Fail":
        badge_html = "<span style='color: red; font-weight: bold;'>FAIL</span>"
    
    title = (
        f"{score_emoji}  **{row['filename']}**"
        f"  —  Score: **{score} / 100** {badge_html}"
        f"  —  Agent: {row['agent_id']}"
        f"  —  {row['lead_source']}"
    )
    with st.expander(title):
        # Display Red Flags alert if any were triggered
        rf_trig = row.get("red_flags_triggered", "")
        if isinstance(rf_trig, list):
            red_flags_triggered = rf_trig
        elif isinstance(rf_trig, str) and rf_trig.strip():
            if rf_trig.startswith("["):
                try:
                    import ast
                    red_flags_triggered = ast.literal_eval(rf_trig)
                except Exception:
                    red_flags_triggered = [x.strip() for x in rf_trig.replace("[","").replace("]","").replace("'","").replace('"',"").split(",") if x.strip()]
            else:
                red_flags_triggered = [x.strip() for x in rf_trig.split(",") if x.strip()]
        else:
            red_flags_triggered = []
            
        red_flag_deduction = row.get("red_flag_deduction", 0)
        try:
            red_flag_deduction = int(red_flag_deduction)
        except Exception:
            red_flag_deduction = 0
            
        if red_flags_triggered and red_flag_deduction != 0:
            st.error(f"⚠️ **Red Flag(s) Triggered:** {', '.join(red_flags_triggered)} ({red_flag_deduction} pts deducted)")

        st.markdown("### Summary")
        st.write(row["summary"])
        st.markdown("### Coaching Points")
        coaching_text = row.get("coaching", row.get("coaching_notes", ""))
        st.write(coaching_text)
        
        st.markdown("### Parameter Scores")
        if parameter_scores:
            for key, p in parameter_scores.items():
                st.markdown(
                    f"- **{p['name']}** ({p['points_max']} pts max): "
                    f"**{p['verdict']}** ({p['points_earned']}/{p['points_max']} pts) "
                    f"— *{p['reason']}*"
                )
        else:
            all_params = get_all_parameter_keys()
            for col in row.index if hasattr(row, "index") else row.keys():
                if col in all_params:
                    val = row[col]
                    if pd.notna(val) and str(val).strip():
                        display_name = col
                        max_points = ""
                        for r in RUBRICS.values():
                            for p in r["parameters"]:
                                if p["key"] == col:
                                    display_name = p["name"]
                                    max_points = f" ({p['max_points']} pts)"
                                    break
                        st.markdown(f"- **{display_name}**{max_points}: {val}")

        if english_transcript and str(english_transcript).strip():
            st.markdown("### 📄 English Transcript")
            formatted_en = str(english_transcript).replace("\n", "  \n")
            st.markdown(formatted_en)

        if transcript_text and str(transcript_text).strip():
            with st.expander("View original transcript (untranslated)"):
                st.text(transcript_text)

        st.caption(
            f"Language detected: {row.get('detected_language', 'unknown')}"
            f"  •  Scored using: {row.get('model_used', 'unknown')}"
        )


# ============================================================
# UI
# ============================================================

st.set_page_config(
    page_title="TSC Call Audit",
    page_icon="📞",
    layout="wide",
)

if "stop_batch_flag" not in st.session_state:
    st.session_state["stop_batch_flag"] = False

st.title("📞 TSC Call Audit Dashboard")
st.caption("AI-scored call quality reports for The Sleep Company")

# Upload section
st.divider()
st.subheader("📤 Score New Calls")
st.caption("Drop audio files or a CSV containing recording URLs — or a mix of both.")

uploaded_files = st.file_uploader(
    "Drop audio files (.mp3/.wav/.m4a/.mpeg/.mp4/.ogg/.flac/.webm) OR a CSV with recording URLs — or a mix of both",
    type=["mp3", "wav", "m4a", "mpeg", "mp4", "ogg", "flac", "webm", "csv"],
    accept_multiple_files=True,
    key="unified_drop",
)

# Helper function to check URL column
def check_url_column(df, col):
    non_empty = df[col].dropna()
    non_empty = [str(val).strip() for val in non_empty if str(val).strip()]
    if not non_empty:
        return False
    check_vals = non_empty[:3]
    for val in check_vals:
        if not (val.startswith("http://") or val.startswith("https://")):
            return False
    return True

# Helper function to find best URL column
def find_best_url_column(qualifying_cols):
    if not qualifying_cols:
        return None
    best_col = None
    best_rank = -1
    for col in qualifying_cols:
        c = col.lower()
        if c == "recording url":
            rank = 10
        elif c == "call recording url":
            rank = 9
        elif c == "url":
            rank = 8
        elif c == "recording":
            rank = 7
        elif "recording url" in c:
            rank = 6
        elif "call recording url" in c:
            rank = 5
        elif "url" in c:
            rank = 4
        elif "recording" in c:
            rank = 3
        else:
            rank = 2
            
        if rank > best_rank:
            best_rank = rank
            best_col = col
    return best_col

csv_files = []
audio_files = []
if uploaded_files:
    for f in uploaded_files:
        if f.name.lower().endswith(".csv"):
            csv_files.append(f)
        else:
            audio_files.append(f)

if len(csv_files) > 1:
    st.error("Only one CSV file can be processed at a time. Ignoring all CSV uploads.")
    csv_files = []

csv_mode = len(csv_files) == 1
csv_df = None
detected_url_col = None
detected_ls_col = None
detected_agent_col = None

if csv_mode:
    csv_file = csv_files[0]
    try:
        csv_bytes = csv_file.getvalue()
        csv_df = pd.read_csv(io.BytesIO(csv_bytes))
        csv_df.columns = [str(c).strip() for c in csv_df.columns]
        
        st.markdown(f"### CSV File: {csv_file.name}")
        st.dataframe(csv_df.head(50), use_container_width=True, hide_index=True)
        st.info(f"Detected {len(csv_df)} rows in {csv_file.name}.")
        
        # URL detection
        qualifying = []
        for col in csv_df.columns:
            col_lower = col.lower()
            if "url" in col_lower or "recording" in col_lower:
                if check_url_column(csv_df, col):
                    qualifying.append(col)
        detected_url_col = find_best_url_column(qualifying)
        
        if detected_url_col:
            st.success(f"🔗 URL column detected: `{detected_url_col}`")
        else:
            st.error("❌ No URL column detected. CSV must have a column with audio recording URLs.")
            
        # Lead source detection
        for col in csv_df.columns:
            if col.lower() in ("lead_source", "lead source"):
                valid_count = sum(1 for val in csv_df[col].dropna() if str(val).strip() in RUBRICS)
                if valid_count > 0:
                    detected_ls_col = col
                    break
        if detected_ls_col:
            st.success(f"🏷️ Lead source column detected: `{detected_ls_col}`")
        else:
            st.info("🏷️ No lead source column — every row will be auto-classified by the AI (per the 'Call type' setting below).")
            
        # Agent ID detection
        for col in csv_df.columns:
            if col.lower() in ("agent_id", "agent id"):
                detected_agent_col = col
                break
    except Exception as e:
        st.error(f"Error parsing CSV file: {e}")
        csv_df = None

# INPUTS
agent_label = "Agent ID (applies to all rows)"
if csv_mode and detected_agent_col:
    agent_label = "Agent ID fallback (used only for rows missing this in the CSV)"

entered_agent_id = st.text_input(
    agent_label,
    placeholder="optional",
    value="",
)

max_rows = 0
if csv_mode:
    max_rows = st.number_input(
        "Max rows to score (0 = all rows)",
        min_value=0,
        value=0,
        step=1,
    )

# CALL TYPE RADIO
radio_label = "Call type"
if csv_mode:
    radio_label = f"Call type (for {csv_files[0].name})"

call_type_choice = st.radio(
    radio_label,
    options=["🧑 Human agent", "🤖 AI voice bot"],
    index=0,
    help="Human agent = uses TSC sales rep rubrics (auto-classified across lead sources unless overridden). AI voice bot = uses the AI Voice Bot rubric for all calls.",
    horizontal=True,
)

# CONDITIONAL DROPDOWN
human_lead_source = AUTO_DETECT_KEY
if call_type_choice == "🧑 Human agent":
    def format_human_ls(k):
        if k == AUTO_DETECT_KEY:
            return "🤖 Auto-detect from audio"
        return RUBRICS[k]["name"]
        
    human_lead_source = st.selectbox(
        "Lead source",
        options=[AUTO_DETECT_KEY] + [k for k in RUBRICS if k != "ai_voice_bot"],
        format_func=format_human_ls,
    )
    
    with st.expander("➕ Add Custom Lead Source + Rubric (Human agent only)", expanded=False):
        st.markdown("Create a new scoring rubric for a custom lead source.")
        
        custom_name = st.text_input("Lead Source Name (display)", placeholder="e.g. Black Friday Campaign")
        derived = derive_key(custom_name)
        if custom_name:
            st.caption(f"Internal key: `{derived}`")
            
        custom_desc = st.text_area("When should this rubric apply? (used by auto-classifier)", placeholder="e.g. Calls about Black Friday offers...")
        
        st.markdown("**Scoring parameters** (total must sum to 100)")
        
        if "custom_params" not in st.session_state:
            st.session_state["custom_params"] = [{"id": 0, "name": "", "points": 10, "desc": ""}]
            st.session_state["param_counter"] = 1
            
        params_to_remove = []
        total_points = 0
        
        for i, p in enumerate(st.session_state["custom_params"]):
            pid = p["id"]
            col1, col2, col3, col4 = st.columns([3, 1, 4, 1])
            p["name"] = col1.text_input("Parameter name", value=p["name"], key=f"pname_{pid}", label_visibility="collapsed", placeholder="Name")
            p["points"] = col2.number_input("Points", min_value=0, max_value=100, value=p["points"], key=f"ppts_{pid}", label_visibility="collapsed")
            p["desc"] = col3.text_input("What to look for", value=p["desc"], key=f"pdesc_{pid}", label_visibility="collapsed", placeholder="Description")
            if col4.button("×", key=f"pdel_{pid}"):
                params_to_remove.append(i)
            total_points += p["points"]
            
        for i in reversed(params_to_remove):
            st.session_state["custom_params"].pop(i)
            st.rerun()
            
        if st.button("+ Add parameter"):
            st.session_state["custom_params"].append({"id": st.session_state["param_counter"], "name": "", "points": 10, "desc": ""})
            st.session_state["param_counter"] += 1
            st.rerun()
            
        if total_points != 100:
            st.warning(f"Total: {total_points} / 100")
        else:
            st.success(f"Total: {total_points} / 100")
            
        with st.expander("Red flags (optional — these deduct points)"):
            if "custom_red_flags" not in st.session_state:
                st.session_state["custom_red_flags"] = []
                st.session_state["rf_counter"] = 0
                
            rf_to_remove = []
            for i, rf in enumerate(st.session_state["custom_red_flags"]):
                rfid = rf["id"]
                c1, c2, c3 = st.columns([3, 1, 1])
                rf["desc"] = c1.text_input("Red flag description", value=rf["desc"], key=f"rfdesc_{rfid}", label_visibility="collapsed", placeholder="Description")
                rf["penalty"] = c2.number_input("Deduction (negative)", max_value=0, value=rf["penalty"], key=f"rfpen_{rfid}", label_visibility="collapsed")
                if c3.button("×", key=f"rfdel_{rfid}"):
                    rf_to_remove.append(i)
                    
            for i in reversed(rf_to_remove):
                st.session_state["custom_red_flags"].pop(i)
                st.rerun()
                
            if st.button("+ Add red flag"):
                st.session_state["custom_red_flags"].append({"id": st.session_state["rf_counter"], "desc": "", "penalty": -10})
                st.session_state["rf_counter"] += 1
                st.rerun()
                
        if st.button("💾 Save Custom Rubric"):
            if not custom_name.strip():
                st.error("Name is required.")
            elif not custom_desc.strip():
                st.error("Description is required.")
            else:
                valid_params = [p for p in st.session_state["custom_params"] if p["name"].strip()]
                if not valid_params:
                    st.error("At least 1 parameter with a name is required.")
                else:
                    new_rubric = {
                        "name": custom_name.strip(),
                        "description": custom_desc.strip(),
                        "parameters": [
                            {"name": p["name"].strip(), "max_points": p["points"], "description": p["desc"].strip()}
                            for p in valid_params
                        ],
                        "red_flags": [
                            {"name": rf["desc"].strip(), "deduction": rf["penalty"], "description": rf["desc"].strip()}
                            for rf in st.session_state["custom_red_flags"] if rf["desc"].strip()
                        ]
                    }
                    save_custom_rubric(derived, new_rubric)
                    st.success("Saved! Refresh the page to see it in the lead source list.")
                    
        st.markdown("**Existing custom rubrics:**")
        crubrics = load_and_convert_custom_rubrics()
        if not crubrics:
            st.caption("None yet.")
        for k, r in crubrics.items():
            col_name, col_del = st.columns([5, 1])
            col_name.write(f"• {r['name']}")
            if col_del.button("🗑️", key=f"del_{k}"):
                delete_custom_rubric(k)
                st.rerun()

    # Feature 3 Preview Table
    if csv_mode and detected_url_col:
        st.markdown("### Lead Source Classification Preview")
        N_preview = min(max_rows, len(csv_df)) if max_rows > 0 else len(csv_df)
        
        state_key = f"preview_state_{len(csv_df)}_{max_rows}"
        if st.session_state.get("current_preview_key") != state_key:
            st.session_state["current_preview_key"] = state_key
            st.session_state["preview_jobs"] = {i: "pending..." for i in range(N_preview)}
            st.session_state["preview_overrides"] = {i: "Use auto-detected" for i in range(N_preview)}
            st.session_state["preview_thread"] = None
            
        def start_preview_classifications(df, url_col, N_prev):
            if st.session_state.get("preview_thread") and st.session_state["preview_thread"].is_alive():
                return
            def bg_task():
                from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
                ctx = get_script_run_ctx()
                with ThreadPoolExecutor(max_workers=3) as pool:
                    for i in range(N_prev):
                        if st.session_state["preview_jobs"].get(i) != "pending...":
                            continue
                        url = str(df.iloc[i][url_col])
                        def task(idx=i, u=url):
                            if ctx:
                                add_script_run_ctx(threading.current_thread(), ctx)
                            try:
                                file_bytes, file_name = fetch_audio_from_url(u)
                                orig_text, lang, eng_text = transcribe_audio(file_bytes, file_name)
                                st.session_state[f"transcript_cache_{idx}"] = {
                                    "orig": orig_text,
                                    "lang": lang,
                                    "eng": eng_text,
                                    "bytes": file_bytes,
                                    "filename": file_name
                                }
                                res = classify_transcript(eng_text)
                                st.session_state["preview_jobs"][idx] = res
                            except Exception as e:
                                st.session_state["preview_jobs"][idx] = "find_a_store"
                        pool.submit(task)
            t = threading.Thread(target=bg_task)
            from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
            ctx = get_script_run_ctx()
            if ctx:
                add_script_run_ctx(t, ctx)
            st.session_state["preview_thread"] = t
            t.start()

        start_preview_classifications(csv_df, detected_url_col, N_preview)
        
        pending_count = sum(1 for v in st.session_state["preview_jobs"].values() if v == "pending...")
        
        # Render the table headers
        tcol1, tcol2, tcol3, tcol4 = st.columns([1, 4, 3, 3])
        tcol1.markdown("**#**")
        tcol2.markdown("**Filename / URL**")
        tcol3.markdown("**Auto-detected**")
        tcol4.markdown("**Override**")
        
        override_options = ["Use auto-detected"] + [k for k in RUBRICS if k != "ai_voice_bot"]
        
        render_limit = min(N_preview, 100)
        for i in range(render_limit):
            col1, col2, col3, col4 = st.columns([1, 4, 3, 3])
            url = str(csv_df.iloc[i][detected_url_col])
            display_url = url if len(url) < 40 else url[:37] + "..."
            auto_val = st.session_state["preview_jobs"].get(i, "pending...")
            if auto_val in RUBRICS:
                auto_val = RUBRICS[auto_val]["name"]
                
            col1.write(f"{i+1}")
            col2.write(display_url)
            col3.write(auto_val)
            
            def make_on_change(idx):
                def cb():
                    st.session_state["preview_overrides"][idx] = st.session_state[f"ovr_key_{idx}"]
                return cb
                
            col4.selectbox(
                "Override",
                options=override_options,
                format_func=lambda x: RUBRICS[x]["name"] if x in RUBRICS else x,
                index=override_options.index(st.session_state["preview_overrides"].get(i, "Use auto-detected")),
                key=f"ovr_key_{i}",
                label_visibility="collapsed",
                on_change=make_on_change(i)
            )
            
        if N_preview > 100:
            st.info(f"Showing first 100 rows of {N_preview} total to maintain UI performance.")
            
        if pending_count > 0:
            time.sleep(1.0)
            st.rerun()


# Compute N
N = 0
score_disabled = True

if csv_mode:
    if csv_df is not None:
        total_rows = len(csv_df)
        if max_rows > 0:
            N = min(max_rows, total_rows)
        else:
            N = total_rows
        if detected_url_col:
            score_disabled = False
else:
    N = len(audio_files)
    if N > 0:
        score_disabled = False

# Render buttons
col_score, col_stop = st.columns([1, 1])
with col_score:
    score_clicked = st.button(f"Score {N} call(s)", type="primary", disabled=score_disabled)
with col_stop:
    stop_clicked = st.button("Stop batch")

if stop_clicked:
    st.session_state["stop_batch_flag"] = True
    st.warning("Stop batch signal sent. Worker threads will terminate on next checkpoint.")

if score_clicked:
    st.session_state["stop_batch_flag"] = False
    
    # Determine source priority
    if csv_mode and detected_url_col:
        source_type = "CSV"
        if len(audio_files) > 0:
            st.warning("⚠️ Both CSV and audio files uploaded. Processing CSV URLs and ignoring uploaded audio files.")
    elif len(audio_files) > 0:
        source_type = "AUDIO"
    else:
        source_type = None

    if source_type:
        progress_bar = st.progress(0.0, text=f"Starting batch of {N} (up to {MAX_PARALLEL} in parallel)...")
        status_placeholder = st.empty()
        succeeded = []
        failed = []
        status_log_lines = []

        def render_status():
            remaining = N - len(succeeded) - len(failed)
            lines = list(status_log_lines)
            if remaining > 0:
                lines.append(f"⏳ _{remaining} file(s) still processing..._")
            status_placeholder.markdown("\n\n".join(lines))

        # Build jobs
        jobs = []
        if source_type == "CSV" and csv_df is not None:
            rows_to_process = csv_df.iloc[:N]
            for i, (_, row_data) in enumerate(rows_to_process.iterrows()):
                # Effective lead source
                if call_type_choice == "🤖 AI voice bot":
                    effective_lead_source = "ai_voice_bot"
                else:
                    override_val = st.session_state.get("preview_overrides", {}).get(i, "Use auto-detected")
                    if override_val != "Use auto-detected":
                        effective_lead_source = override_val
                    else:
                        preview_val = st.session_state.get("preview_jobs", {}).get(i, "pending...")
                        if preview_val != "pending...":
                            effective_lead_source = preview_val
                        elif human_lead_source == AUTO_DETECT_KEY:
                            val = None
                            if detected_ls_col and pd.notna(row_data.get(detected_ls_col)):
                                val_str = str(row_data[detected_ls_col]).strip()
                                if val_str in RUBRICS:
                                    val = val_str
                            effective_lead_source = val if val else AUTO_DETECT_KEY
                        else:
                            effective_lead_source = human_lead_source
                
                # Effective agent id
                effective_agent_id = entered_agent_id
                if detected_agent_col and pd.notna(row_data.get(detected_agent_col)):
                    val_str = str(row_data[detected_agent_col]).strip()
                    if val_str:
                        effective_agent_id = val_str
                        
                has_csv_ls = detected_ls_col and pd.notna(row_data.get(detected_ls_col)) and str(row_data[detected_ls_col]).strip() in RUBRICS
                has_csv_agent = detected_agent_col and pd.notna(row_data.get(detected_agent_col)) and str(row_data[detected_agent_col]).strip()
                csv_driven_flag = True if (has_csv_ls or has_csv_agent) else False
                
                url_val = str(row_data[detected_url_col]).strip()
                
                jobs.append({
                    "source": "url",
                    "url": url_val,
                    "lead_source": effective_lead_source,
                    "agent_id": effective_agent_id,
                    "csv_driven_flag": csv_driven_flag,
                    "row_index": i,
                })
        elif source_type == "AUDIO":
            for file in audio_files:
                if call_type_choice == "🤖 AI voice bot":
                    effective_lead_source = "ai_voice_bot"
                else:
                    effective_lead_source = human_lead_source
                    
                jobs.append({
                    "source": "upload",
                    "bytes": file.getvalue(),
                    "name": file.name,
                    "lead_source": effective_lead_source,
                    "agent_id": entered_agent_id,
                    "csv_driven_flag": False,
                })

        # Submit jobs to ThreadPoolExecutor
        from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
        ctx = get_script_run_ctx()

        def job_wrapper(job):
            if ctx:
                add_script_run_ctx(threading.current_thread(), ctx)
                
            if st.session_state.get("stop_batch_flag"):
                raise RuntimeError("Stopped by user")
                
            cached_data = None
            if job["source"] == "url":
                idx = job.get("row_index")
                if idx is not None and f"transcript_cache_{idx}" in st.session_state:
                    cached_data = st.session_state[f"transcript_cache_{idx}"]
                    file_bytes = cached_data["bytes"]
                    file_name = cached_data["filename"]
                else:
                    file_bytes, file_name = fetch_audio_from_url(job["url"])
            else:
                file_bytes = job["bytes"]
                file_name = job["name"]
                
            if st.session_state.get("stop_batch_flag"):
                raise RuntimeError("Stopped by user")
                
            return process_one_file(
                file_name,
                file_bytes,
                job["lead_source"],
                job["agent_id"],
                csv_driven_flag=job.get("csv_driven_flag", False),
                cached_data=cached_data
            )

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
            futures = {
                executor.submit(job_wrapper, job): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                
                derived_name = job.get("name")
                if not derived_name and job.get("url"):
                    from urllib.parse import urlparse, unquote
                    try:
                        path = unquote(urlparse(job["url"]).path)
                        derived_name = path.rsplit("/", 1)[-1] or "audio.mp3"
                        derived_name = derived_name.split("?")[0]
                        if "." not in derived_name:
                            derived_name += ".mp3"
                    except Exception:
                        derived_name = "audio.mp3"

                try:
                    result_info = future.result()
                    succeeded.append(result_info)
                    auto_tag = " (auto)" if result_info["auto_detected"] else ""
                    csv_tag = " (from CSV)" if result_info.get("csv_driven_flag") else ""
                    status_log_lines.append(
                        f"✅ **{result_info['filename']}** → {result_info['score']}/100  "
                        f"— {result_info['lead_source']}{auto_tag}{csv_tag}"
                    )
                except Exception as e:
                    err_msg = str(e)
                    if "Stopped by user" in err_msg:
                        failed.append({"filename": derived_name, "error": "cancelled (stop pressed)"})
                        status_log_lines.append(f"🛑 **{derived_name}** → cancelled (stop pressed)")
                    else:
                        failed.append({"filename": derived_name, "error": err_msg})
                        status_log_lines.append(f"❌ **{derived_name}** → failed: {err_msg[:100]}")
                
                done = len(succeeded) + len(failed)
                progress_bar.progress(done / N, text=f"Completed {done} of {N}")
                render_status()

        # Reset stop flag after batch completes/stops
        st.session_state["stop_batch_flag"] = False

        if failed and succeeded:
            st.warning(f"⚠️ Batch done. {len(succeeded)} succeeded, {len(failed)} failed.")
        elif failed and not succeeded:
            st.error(f"❌ Batch done. All {len(failed)} file(s) failed.")
        else:
            st.success(f"✅ Batch done. Scored {len(succeeded)} call(s) and appended to today's report.")

        if succeeded:
            st.markdown("### Batch Summary")
            summary_df = pd.DataFrame([
                {
                    "Filename": s["filename"],
                    "Lead Source": s["lead_source"] + (" (auto)" if s["auto_detected"] else ""),
                    "Score": s["score"],
                }
                for s in succeeded
            ])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        if failed:
            st.markdown("### Failures")
            for f in failed:
                st.markdown(f"- ❌ **{f['filename']}** — {f['error']}")

        if succeeded:
            st.markdown("### Detailed Results")
            for s in succeeded:
                render_call_card(
                    s["row"],
                    transcript_text=s["transcript"],
                    english_transcript=s.get("english_transcript"),
                    parameter_scores=s.get("parameter_scores")
                )
                with st.expander(f"View transcript — {s['filename']}"):
                    st.text(s["transcript"])

# Reports section
st.divider()
st.subheader("📊 Audit Reports")

output_dir = Path("output")
csv_files = sorted(output_dir.glob("audit-report-*.csv"), reverse=True)

if not csv_files:
    st.info(
        "No audit reports yet. Upload calls above, or run `python pipeline.py` "
        "in your terminal to bulk-process recordings."
    )
else:
    selected_report = st.selectbox(
        "Select an audit report",
        csv_files,
        format_func=lambda p: p.name,
        key="report_selector",
    )
    df = pd.read_csv(selected_report, encoding="utf-8")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Calls", len(df))
    col2.metric("Average Score", f"{df['total_score'].mean():.1f} / 100")
    col3.metric("Highest Score", f"{int(df['total_score'].max())} / 100")
    col4.metric("Lowest Score", f"{int(df['total_score'].min())} / 100")
    st.divider()
    st.markdown("**Calls** — sorted by score, lowest first (these need coaching most)")
    df_sorted = df.sort_values("total_score", ascending=True).reset_index(drop=True)
    for _, row in df_sorted.iterrows():
        orig_t, en_t = load_transcripts_from_disk(row.get("filename"))
        render_call_card(row, transcript_text=orig_t, english_transcript=en_t)
