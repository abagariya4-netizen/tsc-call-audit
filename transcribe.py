import os
from dotenv import load_dotenv
from groq import Groq
import deterministic_cache as dc

# Load secrets from the .env file into our environment
load_dotenv()

# Grab the Groq API key we stored in .env
api_key = os.getenv("GROQ_API_KEY")

# Tell Groq "hi, here's who I am" — sets up the connection
client = Groq(api_key=api_key)

# Path to the audio file we want to transcribe
audio_path = "recordings/test-call-1.mpeg"

if not os.path.exists(audio_path):
    print(f"Audio file {audio_path} not found.")
else:
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    h = dc.audio_hash(audio_bytes)
    cached = dc.cache_get("transcripts", h)

    if cached:
        print("--- Transcript cache hit! ---")
        transcript_text = cached["transcript"]
        detected_language = cached["language"]
    else:
        print(f"Transcribing {audio_path} ... this can take 10-60 seconds.")
        # Send it to Groq's Whisper model for transcription with temperature=0.0
        transcript = client.audio.transcriptions.create(
            file=(audio_path, audio_bytes),
            model="whisper-large-v3",
            response_format="verbose_json",
            temperature=0.0,
        )
        transcript_text = transcript.text
        detected_language = getattr(transcript, "language", "")
        
        # Save to cache
        dc.cache_set("transcripts", h, {
            "transcript": transcript_text,
            "language": detected_language,
            "english": transcript_text
        })

    # Print what came back
    print("\n" + "=" * 60)
    print("DETECTED LANGUAGE:", detected_language)
    print("=" * 60)
    print(transcript_text)
    print("=" * 60)