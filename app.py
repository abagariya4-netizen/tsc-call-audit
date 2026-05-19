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

UPLOAD_CHOICES = [AUTO_DETECT_KEY] + LEAD_SOURCE_KEYS

# Call-type radio keys
CALL_TYPE_HUMAN = "human"
CALL_TYPE_BOT = "bot"

# Human-only rubrics (subset of RUBRICS keys, excluding ai_voice_bot)
HUMAN_LEAD_SOURCE_KEYS = [k for k in LEAD_SOURCE_KEYS if k != "ai_voice_bot"]

# Per-file dropdown choices when call type is HUMAN
HUMAN_UPLOAD_CHOICES = [AUTO_DETECT_KEY] + HUMAN_LEAD_SOURCE_KEYS

MAX_PARALLEL = 20

META_COLS = {
    "filename", "agent_id", "lead_source", "detected_language",
    "total_score", "red_flags_triggered", "summary", "coaching_notes",
    "model_used",
}

# ============================================================
# HELPERS
# ============================================================

def call_with_retry(fn, *args, max_retries: int = 4, **kwargs):
    """
    Call fn(*args, **kwargs). If it raises a rate-limit error (429 / quota
    exceeded / "rate limit" / "resource exhausted" in the message), wait
    with exponential backoff and retry. Other errors propagate immediately.
    """
    import re
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = (
                "429" in msg
                or "rate limit" in msg
                or "rate_limit" in msg
                or "resource_exhausted" in msg
                or "resourceexhausted" in msg
                or "quota" in msg
                or "too many requests" in msg
            )
            if not is_rate_limit:
                raise  # non-retriable, fail fast
            last_err = e
            # Try to extract a "retry after N seconds" hint from the error
            wait_seconds = None
            m = re.search(r"retry in ([\d.]+)s", msg)
            if m:
                try:
                    wait_seconds = float(m.group(1))
                except ValueError:
                    pass
            if wait_seconds is None:
                # Exponential backoff with jitter: 5s, 10s, 20s, 40s
                wait_seconds = 5 * (2 ** attempt)
            # Cap at 60s so one slow call doesn't stall everything forever
            wait_seconds = min(wait_seconds, 60.0)
            time.sleep(wait_seconds)
    # Out of retries
    raise RuntimeError(
        f"Rate-limited after {max_retries} retries. Last error: {last_err}"
    ) from last_err

def format_lead_source_choice(k):
    if k == AUTO_DETECT_KEY:
        return "🤖 Auto-detect from audio"
    return RUBRICS[k]['name']


def transcribe_audio(file_bytes, filename):
    transcript = call_with_retry(
        groq_client.audio.transcriptions.create,
        file=(filename, file_bytes),
        model="whisper-large-v3",
        response_format="verbose_json",
    )
    translation = call_with_retry(
        groq_client.audio.translations.create,
        file=(filename, file_bytes),
        model="whisper-large-v3",
        response_format="verbose_json",
    )
    return transcript.text, transcript.language, translation.text


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

Return ONLY valid JSON in this exact structure:

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
    last_error = None
    for model_name in MODEL_FALLBACK_LADDER:
        try:
            response = call_with_retry(
                gemini_client.models.generate_content,
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


def auto_classify_human_and_score(transcript_text):
    """Single Gemini call: classify across the 6 HUMAN lead sources only
    (no ai_voice_bot) AND score against the matched rubric."""
    all_rubrics_text = ""
    for key in HUMAN_LEAD_SOURCE_KEYS:
        all_rubrics_text += f"\n\n=== LEAD SOURCE KEY: {key} ===\n"
        all_rubrics_text += format_rubric_for_prompt(key)

    prompt = f"""You are a strict call-quality auditor for TSC. Two-step task:

STEP 1: Identify the lead source from these 6 HUMAN agent categories:
- find_a_store: customer searched WhatsApp for nearest store
- arrange_callback: customer requested callback from website
- inbound: customer called us
- shopflo_abandoned_cart: customer added to cart but didn't buy
- next_day_delivery: customer wants next-day delivery
- no_cost_emi: customer wants EMI

STEP 2: Score against the matching rubric.

ALL HUMAN RUBRICS:
{all_rubrics_text}

TRANSCRIPT:
---
{transcript_text}
---

Return ONLY valid JSON:
{{
  "lead_source": "<key>",
  "parameters": [{{"name": "...", "score": 0, "max": 0, "reason": "..."}}],
  "red_flags_triggered": [{{"description": "...", "penalty": 0}}],
  "total_score": 0,
  "summary": "...",
  "coaching_notes": "..."
}}

The "parameters" and "red_flags_triggered" MUST match the rubric of the
identified lead_source.
"""
    last_error = None
    for model_name in MODEL_FALLBACK_LADDER:
        try:
            response = call_with_retry(
                gemini_client.models.generate_content,
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            result = json.loads(response.text)
            lead_source = result.get("lead_source", "find_a_store")
            if lead_source not in HUMAN_LEAD_SOURCE_KEYS:
                lead_source = "find_a_store"
            return result, model_name, lead_source
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All Gemini models failed in auto-classify (human). Last error: {last_error}")


def auto_classify_and_score(transcript_text):
    """Single Gemini call that BOTH classifies the lead source AND scores against the matching rubric.
    Returns (result_dict, model_used, lead_source)."""
    all_rubrics_text = ""
    for key, data in RUBRICS.items():
        all_rubrics_text += f"\n\n=== LEAD SOURCE KEY: {key} ===\n"
        all_rubrics_text += format_rubric_for_prompt(key)

    prompt = f"""You are a strict call-quality auditor for The Sleep Company (TSC),
a premium mattress brand in India. You audit calls made by outbound and inbound
sales agents to leads. The calls are typically in Hindi, English, or other
Indian languages — work based on meaning, not language.

TASK — TWO STEPS:

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

ALL 6 RUBRICS:
{all_rubrics_text}

CALL TRANSCRIPT (single-speaker raw output; infer agent vs customer):
---
{transcript_text}
---

Return ONLY valid JSON in this EXACT structure:

{{
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
            response = call_with_retry(
                gemini_client.models.generate_content,
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            result = json.loads(response.text)
            lead_source = result.get("lead_source", "find_a_store")
            if lead_source not in RUBRICS:
                lead_source = "find_a_store"
            return result, model_name, lead_source
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All Gemini models failed in auto-classify. Last error: {last_error}")


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


def save_recording_and_transcript(file_bytes, original_filename, transcript_text, language, english_translation):
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
    with open(Path("transcripts") / f"{safe_stem}.en.txt", "w", encoding="utf-8") as f:
        f.write(english_translation)
    return safe_filename


def derive_filename_from_url(url: str) -> str:
    from urllib.parse import urlparse, unquote
    import hashlib
    path = urlparse(url).path
    name = unquote(path.split("/")[-1]) if path else ""
    if not name or len(name) > 100:
        name = f"csv-row-{hashlib.md5(url.encode()).hexdigest()[:10]}.mp4"
    if "." not in name:
        name += ".mp4"
    return name

def download_audio(url: str, timeout: int = 60) -> bytes:
    import requests
    last_err = None
    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise RuntimeError(f"Failed to download after 2 attempts: {last_err}")


def process_one_file(uploaded_name, uploaded_bytes, lead_source_or_auto, agent_id, call_type):
    """Full pipeline for one file. Designed to run in a worker thread."""
    transcript_text, detected_language, english_translation = transcribe_audio(uploaded_bytes, uploaded_name)

    if call_type == CALL_TYPE_BOT:
        # Skip auto-detect entirely — force ai_voice_bot
        lead_source = "ai_voice_bot"
        result, model_used = score_transcript(transcript_text, lead_source)
        auto_detected = False
    elif lead_source_or_auto == AUTO_DETECT_KEY:
        # Human auto-detect across 6 human rubrics only
        result, model_used, lead_source = auto_classify_human_and_score(transcript_text)
        auto_detected = True
    else:
        # Manual per-file choice (must be a human rubric here)
        lead_source = lead_source_or_auto
        result, model_used = score_transcript(transcript_text, lead_source)
        auto_detected = False

    safe_filename = save_recording_and_transcript(
        uploaded_bytes, uploaded_name, transcript_text, detected_language, english_translation
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
        "row": display_row,
        "transcript": transcript_text,
        "english_translation": english_translation,
    }


def render_call_card(row):
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

def check_password():
    """Render a password gate. Halts the app until the correct password is entered."""
    correct_password = os.getenv("APP_PASSWORD", "")
    if not correct_password or correct_password == "changeme-set-a-real-password-here":
        st.error(
            "🔒 APP_PASSWORD is not set in your .env file. "
            "Open .env and set APP_PASSWORD to a real password, then restart streamlit."
        )
        st.stop()

    if st.session_state.get("authenticated", False):
        return  # Already unlocked this session

    st.markdown(
        """
        <div style='text-align: center; padding: 80px 20px 20px 20px;'>
            <h1>🔒 TSC Call Audit</h1>
            <p style='color: #6b7280;'>Enter the access password to continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Center the form using columns
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        with st.form("auth_form", clear_on_submit=False):
            pw = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Password")
            submitted = st.form_submit_button("Unlock", type="primary", use_container_width=True)
            if submitted:
                if pw == correct_password:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect password. Try again.")

    st.stop()

check_password()

with st.sidebar:
    st.markdown("### 🔐 Session")
    if st.button("Log out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

st.title("📞 TSC Call Audit Dashboard")
st.caption("AI-scored call quality reports for The Sleep Company")

# Upload section
st.divider()
st.subheader("📤 Score New Calls")
st.caption("Drag and drop one or more call recordings. Lead source is auto-detected by default — override per file if needed.")

audio_call_type = st.radio(
    "Call type",
    options=[CALL_TYPE_HUMAN, CALL_TYPE_BOT],
    format_func=lambda v: "🧑 Human agent" if v == CALL_TYPE_HUMAN else "🤖 AI voice bot",
    horizontal=True,
    key="audio_call_type",
    help="Pick 'AI voice bot' if these recordings are from your outbound voice bot platform. Human agent is the default."
)

uploaded_files = st.file_uploader(
    "Drop audio files here, or click to browse (one or many)",
    type=["mp3", "wav", "m4a", "mpeg", "mp4", "ogg", "flac", "webm"],
    accept_multiple_files=True,
    help="Supports up to ~25MB per file (~30 min of audio). Files are processed in parallel, up to 4 at a time.",
)

entered_agent_id = st.text_input(
    "Agent ID (applies to all files in this batch)",
    placeholder="optional",
    value="",
)

file_lead_sources = {}
if uploaded_files:
    if audio_call_type == CALL_TYPE_BOT:
        st.info(f"🤖 All {len(uploaded_files)} file(s) will be scored against the AI Voice Bot rubric.")
        # Build a dict where every file maps to AUTO_DETECT_KEY — the value is ignored when call_type is BOT
        file_lead_sources = {u.name: AUTO_DETECT_KEY for u in uploaded_files}
    else:
        # Existing human-mode loop, but use HUMAN_UPLOAD_CHOICES (no ai_voice_bot in dropdown)
        st.markdown("**Lead source per file** (default: auto-detect)")
        for uploaded in uploaded_files:
            cols = st.columns([3, 2])
            with cols[0]:
                st.markdown(f"📄 **{uploaded.name}**  _({uploaded.size / 1024:.0f} KB)_")
            with cols[1]:
                file_lead_sources[uploaded.name] = st.selectbox(
                    f"Lead source for {uploaded.name}",
                    HUMAN_UPLOAD_CHOICES,
                    format_func=format_lead_source_choice,
                    key=f"ls_{uploaded.name}_{uploaded.size}",
                    label_visibility="collapsed",
                )

n_files = len(uploaded_files) if uploaded_files else 0
button_label = "Score this call" if n_files <= 1 else f"Score these {n_files} calls"
score_button_disabled = n_files == 0

if st.button(button_label, type="primary", disabled=score_button_disabled):
    batch_start_time = time.time()
    progress_bar = st.progress(0.0, text=f"Starting batch of {n_files} (up to {MAX_PARALLEL} in parallel)...")
    status_placeholder = st.empty()
    succeeded = []
    failed = []
    status_log_lines = []

    def render_status():
        completed = len(succeeded) + len(failed)
        remaining = n_files - completed
        lines = list(status_log_lines)
        
        elapsed = time.time() - batch_start_time
        if completed > 0:
            per_call = elapsed / completed
            remaining_time = (n_files - completed) * per_call
            lines.insert(
                0,
                f"⏱️  Elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s  •  "
                f"ETA: {int(remaining_time // 60)}m {int(remaining_time % 60)}s  •  "
                f"Throughput: {completed / elapsed * 60:.1f} calls/min"
            )

        if remaining > 0:
            lines.append(f"⏳ _{remaining} file(s) still processing..._")
        status_placeholder.markdown("\n\n".join(lines))

    # Prepare jobs (we read bytes here so threads don't share the UploadedFile object)
    jobs = []
    for uploaded in uploaded_files:
        jobs.append({
            "name": uploaded.name,
            "bytes": uploaded.getvalue(),
            "lead_source_choice": file_lead_sources[uploaded.name],
        })

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {
            executor.submit(
                process_one_file,
                job["name"],
                job["bytes"],
                job["lead_source_choice"],
                entered_agent_id,
                audio_call_type,
            ): job["name"]
            for job in jobs
        }
        for future in as_completed(futures):
            filename = futures[future]
            try:
                result_info = future.result()
                succeeded.append(result_info)
                auto_tag = " (auto)" if result_info["auto_detected"] else ""
                status_log_lines.append(
                    f"✅ **{filename}** → {result_info['score']}/100  "
                    f"— {result_info['lead_source']}{auto_tag}"
                )
            except Exception as e:
                failed.append({"filename": filename, "error": str(e)})
                status_log_lines.append(f"❌ **{filename}** → failed: {str(e)[:100]}")
            done = len(succeeded) + len(failed)
            progress_bar.progress(done / n_files, text=f"Completed {done} of {n_files}")
            render_status()

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
            render_call_card(s["row"])
            with st.expander(f"View transcript — {s['filename']}"):
                tab1, tab2 = st.tabs(["Original", "English translation"])
                with tab1:
                    st.text(s["transcript"])
                with tab2:
                    st.text(s["english_translation"])

# ============================================================
# CSV Upload section
# ============================================================
st.divider()
st.subheader("📋 Score from CSV (bulk URL mode)")
st.caption(
    "Upload a CSV containing call recording URLs. The tool downloads each "
    "recording, transcribes, translates, and scores against the matching "
    "rubric. Useful for CRM exports and voice bot platforms."
)

csv_call_type = st.radio(
    "Call type",
    options=[CALL_TYPE_HUMAN, CALL_TYPE_BOT],
    format_func=lambda v: "🧑 Human agent" if v == CALL_TYPE_HUMAN else "🤖 AI voice bot",
    horizontal=True,
    key="csv_call_type",
    help="If this CSV is an export from your voice bot platform, pick 'AI voice bot'. Default is human agent."
)

csv_upload = st.file_uploader(
    "Drop a CSV file here, or click to browse",
    type=["csv"],
    accept_multiple_files=False,
    key="csv_uploader",
)

if csv_upload is not None:
    df = pd.read_csv(csv_upload)
    st.dataframe(df.head(5), use_container_width=True, hide_index=True)
    st.info(f"Detected {len(df)} rows in this CSV.")

    url_col = next((c for c in df.columns if any(sub in c.lower() for sub in ["url", "recording", "audio"])), None)
    if not url_col:
        st.error("No URL column detected. Looking for a column with 'url', 'recording', or 'audio' in its name.")
        st.stop()
    else:
        st.success(f"🔗 URL column detected: `{url_col}`")

    ls_col = next((c for c in df.columns if "lead_source" in c.lower() or "lead source" in c.lower()), None)
    if ls_col:
        st.write(f"🏷️ Lead source column: `{ls_col}`")
    else:
        st.write("🏷️ No lead source column — every row will be auto-classified by the AI.")

    agent_col = next((c for c in df.columns if c.lower() == "agent_id" or ("agent" in c.lower() and "id" in c.lower())), None)
    if agent_col:
        st.write(f"👤 Agent ID column: `{agent_col}`")
        csv_agent_id_fallback = ""
    else:
        csv_agent_id_fallback = st.text_input("Agent ID (applies to all rows)", key="csv_agent_id")

    max_rows = st.number_input(
        "Max rows to score (0 = all rows)",
        min_value=0, value=0, step=10,
        key="csv_max_rows",
        help="0 means no cap — score every row. Set a number to limit "
             "(useful for testing or staying within API quotas)."
    )
    effective_n = len(df) if max_rows == 0 else min(len(df), max_rows)

    if st.button(f"Score {effective_n} call(s) from CSV", type="primary", key="csv_score_btn"):
        st.session_state["csv_stop_flag"] = False
        
        if st.button("🛑 Stop batch", key="csv_stop_btn"):
            st.session_state["csv_stop_flag"] = True
            
        csv_batch_start_time = time.time()
        csv_progress_bar = st.progress(0.0, text=f"Starting CSV batch of {effective_n} (up to {MAX_PARALLEL} in parallel)...")
        csv_status_placeholder = st.empty()
        csv_succeeded = []
        csv_failed = []
        csv_status_log_lines = []
        
        def render_csv_status():
            completed = len(csv_succeeded) + len(csv_failed)
            remaining = effective_n - completed
            lines = list(csv_status_log_lines)
            
            elapsed = time.time() - csv_batch_start_time
            if completed > 0:
                per_call = elapsed / completed
                remaining_time = remaining * per_call
                lines.insert(
                    0,
                    f"⏱️  Elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s  •  "
                    f"ETA: {int(remaining_time // 60)}m {int(remaining_time % 60)}s  •  "
                    f"Throughput: {completed / elapsed * 60:.1f} calls/min"
                )

            if remaining > 0:
                lines.append(f"⏳ _{remaining} file(s) still processing..._")
            csv_status_placeholder.markdown("\n\n".join(lines))
            
        df_to_process = df.head(effective_n)
        
        jobs = []
        for idx, row in df_to_process.iterrows():
            jobs.append({
                "url": row[url_col],
                "lead_source": row[ls_col] if ls_col and pd.notna(row[ls_col]) and row[ls_col] in RUBRICS else AUTO_DETECT_KEY,
                "agent_id": str(row[agent_col]) if agent_col and pd.notna(row[agent_col]) else csv_agent_id_fallback
            })

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
            future_to_job = {}
            for job in jobs:
                if st.session_state.get("csv_stop_flag", False):
                    break
                    
                def process_csv_row(job=job):
                    audio_bytes = download_audio(job["url"])
                    filename = derive_filename_from_url(job["url"])
                    return process_one_file(
                        uploaded_name=filename,
                        uploaded_bytes=audio_bytes,
                        lead_source_or_auto=job["lead_source"],
                        agent_id=job["agent_id"],
                        call_type=csv_call_type,
                    )
                
                future = executor.submit(process_csv_row)
                future_to_job[future] = job
                
            for future in as_completed(future_to_job):
                if st.session_state.get("csv_stop_flag", False):
                    break
                job = future_to_job[future]
                try:
                    result_info = future.result()
                    csv_succeeded.append(result_info)
                    auto_tag = " (auto)" if result_info["auto_detected"] else ""
                    csv_status_log_lines.append(
                        f"✅ **{result_info['filename']}** → {result_info['score']}/100  "
                        f"— {result_info['lead_source']}{auto_tag}"
                    )
                except Exception as e:
                    csv_failed.append({"filename": job["url"], "error": str(e)})
                    csv_status_log_lines.append(f"❌ **{str(job['url'])[:30]}...** → failed: {str(e)[:100]}")
                
                done = len(csv_succeeded) + len(csv_failed)
                csv_progress_bar.progress(done / effective_n, text=f"Completed {done} of {effective_n}")
                render_csv_status()

        if csv_failed and csv_succeeded:
            st.warning(f"⚠️ Batch done. {len(csv_succeeded)} succeeded, {len(csv_failed)} failed.")
        elif csv_failed and not csv_succeeded:
            st.error(f"❌ Batch done. All {len(csv_failed)} file(s) failed.")
        else:
            st.success(f"✅ Batch done. Scored {len(csv_succeeded)} call(s) and appended to today's report.")
            
        if csv_succeeded:
            st.markdown("### Batch Summary")
            csv_summary_df = pd.DataFrame([
                {
                    "Filename": s["filename"],
                    "Lead Source": s["lead_source"] + (" (auto)" if s["auto_detected"] else ""),
                    "Score": s["score"],
                }
                for s in csv_succeeded
            ])
            st.dataframe(csv_summary_df, use_container_width=True, hide_index=True)
            
        if csv_failed:
            st.markdown("### Failures")
            for f in csv_failed:
                st.markdown(f"- ❌ **{f['filename']}** — {f['error']}")
                
        if csv_succeeded:
            st.markdown("### Detailed Results")
            for s in csv_succeeded:
                render_call_card(s["row"])
                with st.expander(f"View transcript — {s['filename']}"):
                    tab1, tab2 = st.tabs(["Original", "English translation"])
                    with tab1:
                        st.text(s["transcript"])
                    with tab2:
                        st.text(s["english_translation"])

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
        render_call_card(row)
