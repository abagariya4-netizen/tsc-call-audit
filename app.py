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

CUSTOM_RUBRICS = load_custom_rubrics()
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

LEAD_SOURCE_KEYS = list(RUBRICS.keys())

AUTO_DETECT_KEY = "__auto__"
AI_CATEGORY_KEYS = {"ai_voice_bot"}

UPLOAD_CHOICES = [AUTO_DETECT_KEY, "ai_voice_bot"]

MAX_PARALLEL = 4

META_COLS = {
    "filename", "agent_id", "lead_source", "detected_language",
    "total_score", "red_flags_triggered", "summary", "coaching_notes",
    "model_used",
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
    transcript = groq_client.audio.transcriptions.create(
        file=(filename, file_bytes),
        model="whisper-large-v3",
        response_format="verbose_json",
    )
    try:
        translation = groq_client.audio.translations.create(
            file=(filename, file_bytes),
            model="whisper-large-v3",
            response_format="verbose_json",
        )
        english_translation = translation.text
    except Exception:
        english_translation = transcript.text
    return transcript.text, transcript.language, english_translation


def score_transcript(transcript_text, lead_source):
    """Single-rubric scoring (used when user manually picks a lead source)."""
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

Generate a naturally translated English version of the transcript, formatted as a dialogue where each turn starts with "Agent:" or "Customer:" on a new line with blank lines between speaker turns. "Agent" is the TSC sales representative; "Customer" is the lead. Translate naturally (not word-for-word); preserve names, prices, store addresses, pincodes, and product names verbatim. If the call is already in English, clean it up (remove filler) and format as dialogue.

Return ONLY valid JSON in this exact structure:

{{
  "english_transcript": "translated and formatted dialogue",
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
    last_error = None
    for model_name in MODEL_FALLBACK_LADDER:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            return json.loads(response.text), model_name
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


def auto_classify_and_score(transcript_text):
    """Single Gemini call that BOTH classifies the lead source AND scores against the matching rubric.
    Returns (result_dict, model_used, lead_source)."""
    all_rubrics_text = ""
    for key, data in RUBRICS.items():
        if key == "ai_voice_bot":
            continue
        all_rubrics_text += f"\n\n=== LEAD SOURCE KEY: {key} ===\n"
        all_rubrics_text += format_rubric_for_prompt(key)

    prompt = f"""You are a strict call-quality auditor for The Sleep Company (TSC),
a premium mattress brand in India. You audit calls made by outbound and inbound
sales agents to leads. The calls are typically in Hindi, English, or other
Indian languages — work based on meaning, not language.

TASK — THREE STEPS:

STEP 1: Identify which of these 6 lead sources this call belongs to, based
on the transcript content:

- find_a_store: HIGH-INTENT customer searched for nearest store on WhatsApp.
  Agent confirms store address/timings, pushes for visit.
- arrange_callback: EXPLORATORY. Customer requested callback from website.
  Agent qualifies needs first, then pitches.
- inbound: HIGHEST INTENT. Customer called us. Agent listens first.
- shopflo_abandoned_cart: HIGH INTENT. Customer added to cart but didn't buy.
  Agent probes the blocker, removes it, closes.
- next_day_delivery: URGENCY. Customer wants next-day delivery.
  Agent confirms pincode, explains free pillow guarantee, closes fast.
- no_cost_emi: PRICE SENSITIVE. Customer wants EMI option.
  Agent explains EMI math clearly, closes.

STEP 2: Apply the matching rubric (from below) to score the call.

STEP 3: Generate a naturally translated English version of the transcript, formatted as a dialogue where each turn starts with "Agent:" or "Customer:" on a new line with blank lines between speaker turns. "Agent" is the TSC sales representative; "Customer" is the lead. Translate naturally (not word-for-word); preserve names, prices, store addresses, pincodes, and product names verbatim. If the call is already in English, clean it up (remove filler) and format as dialogue.

ALL 6 RUBRICS:
{all_rubrics_text}

CALL TRANSCRIPT (single-speaker raw output; infer agent vs customer):
---
{transcript_text}
---

Return ONLY valid JSON in this EXACT structure:

{{
  "english_transcript": "translated and formatted dialogue",
  "lead_source": "<one of: find_a_store, arrange_callback, inbound, shopflo_abandoned_cart, next_day_delivery, no_cost_emi>",
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

The "parameters" and "red_flags_triggered" MUST match the rubric of the
lead_source you identified — do not mix rubrics.
"""
    last_error = None
    for model_name in MODEL_FALLBACK_LADDER:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            result = json.loads(response.text)
            lead_source = result.get("lead_source", "find_a_store")
            if lead_source not in RUBRICS or lead_source == "ai_voice_bot":
                lead_source = "find_a_store"
            return result, model_name, lead_source
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All Gemini models failed in auto-classify. Last error: {last_error}")


def classify_transcript(english_transcript):
    all_keys = [k for k in RUBRICS if k != "ai_voice_bot"]
    desc_list = []
    for k in all_keys:
        desc = RUBRICS[k].get("intent", RUBRICS[k].get("description", ""))
        desc_list.append(f"- {k}: {desc}")
    desc_str = "\n".join(desc_list)
    
    prompt = f"""SYSTEM: You are classifying calls into one of the following lead source categories. Pick exactly ONE category that best fits the transcript.

Categories:
{desc_str}

Transcript:
{english_transcript}

Respond with ONLY the category key, no other text."""

    for model_name in MODEL_FALLBACK_LADDER:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            val = response.text.strip().lower()
            if val in RUBRICS and val != "ai_voice_bot":
                return val
            for k in all_keys:
                if k in val:
                    return k
            return "find_a_store"
        except Exception:
            pass
    return "find_a_store"


def build_csv_row(filename, agent_id, lead_source, language, result, model_used):
    row = {
        "filename": filename,
        "agent_id": agent_id,
        "lead_source": lead_source,
        "detected_language": language,
        "total_score": result.get("total_score", 0),
        "summary": result.get("summary", ""),
        "coaching_notes": result.get("coaching_notes", ""),
        "model_used": model_used,
    }
    for p in result.get("parameters", []):
        name = p.get("name", "")
        score = p.get("score", 0)
        maxv = p.get("max", 0)
        reason = p.get("reason", "")
        if name:
            row[name] = f"{score}/{maxv} - {reason}"
    rfs = result.get("red_flags_triggered", [])
    rf_strs = [
        f"{rf.get('description', '')} ({rf.get('penalty', 0)})"
        for rf in rfs
    ]
    row["red_flags_triggered"] = "; ".join(rf_strs)
    return row


def _do_append_to_today_csv(row_dict):
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / f"audit-report-{today}.csv"
    file_exists = csv_path.exists()

    if file_exists:
        existing_df = pd.read_csv(csv_path, nrows=0)
        columns = list(existing_df.columns)
    else:
        all_params = set()
        for r in RUBRICS.values():
            for p_name, _, _ in r["parameters"]:
                all_params.add(p_name)
        sorted_params = sorted(all_params)
        columns = (
            ["filename", "agent_id", "lead_source", "detected_language", "total_score"]
            + sorted_params
            + ["red_flags_triggered", "summary", "coaching_notes", "model_used"]
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
        result, model_used = score_transcript(transcript_text, lead_source)
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


def render_call_card(row, transcript_text=None, english_transcript=None):
    score = int(row["total_score"])
    if score >= 80:
        score_emoji = "🟢"
    elif score >= 50:
        score_emoji = "🟡"
    else:
        score_emoji = "🔴"
    title = (
        f"{score_emoji}  **{row['filename']}**"
        f"  —  Score: **{score} / 100**"
        f"  —  Agent: {row['agent_id']}"
        f"  —  {row['lead_source']}"
    )
    with st.expander(title):
        st.markdown("### Summary")
        st.write(row["summary"])
        st.markdown("### Coaching Notes")
        st.write(row["coaching_notes"])
        st.markdown("### Parameter Scores")
        any_param = False
        for col in row.keys():
            if col in META_COLS:
                continue
            val = row[col]
            if pd.notna(val) and str(val).strip():
                st.markdown(f"- **{col}**: {val}")
                any_param = True
        if not any_param:
            st.markdown("_No parameter scores recorded._")
        st.markdown("### Red Flags")
        red_flags = row.get("red_flags_triggered", "")
        if pd.notna(red_flags) and str(red_flags).strip():
            for rf in str(red_flags).split(";"):
                rf = rf.strip()
                if rf:
                    st.markdown(f"- ⚠️ {rf}")
        else:
            st.markdown("_None_")
        
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
    help="Human agent = uses TSC sales rep rubrics (auto-classified across 6 lead sources unless overridden). AI voice bot = uses the AI Voice Bot rubric for all calls.",
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
        crubrics = load_custom_rubrics()
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
                render_call_card(s["row"], transcript_text=s["transcript"], english_transcript=s.get("english_transcript"))
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
