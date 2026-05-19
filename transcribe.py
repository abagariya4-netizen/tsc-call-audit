import os
from dotenv import load_dotenv
from groq import Groq

# Load secrets from the .env file into our environment
load_dotenv()

# Grab the Groq API key we stored in .env
api_key = os.getenv("GROQ_API_KEY")

# Tell Groq "hi, here's who I am" — sets up the connection
client = Groq(api_key=api_key)

# Path to the audio file we want to transcribe
audio_path = "recordings/test-call-1.mpeg"

print(f"Transcribing {audio_path} ... this can take 10-60 seconds.")

# Open the audio file in binary mode ("rb" = read binary)
# and send it to Groq's Whisper model for transcription
with open(audio_path, "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        file=(audio_path, audio_file.read()),
        model="whisper-large-v3",
        response_format="verbose_json",
    )

# Print what came back
print("\n" + "=" * 60)
print("DETECTED LANGUAGE:", transcript.language)
print("=" * 60)
print(transcript.text)
print("=" * 60)