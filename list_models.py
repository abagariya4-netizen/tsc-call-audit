import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Models your API key can use for generateContent:\n")
for m in client.models.list():
    if m.supported_actions and "generateContent" in m.supported_actions:
        print(f"  {m.name}")