from groq import Groq
import os
from dotenv import load_dotenv

# Load .env from this file's directory so behavior is stable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")

if not GROQ_API_KEY:
    print("No API key found in environment")
    exit(1)

client = Groq(api_key=GROQ_API_KEY)

try:
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "test"}]
    )
    print("API call successful")
    print(f"Reply: {completion.choices[0].message.content}")
except Exception as e:
    print(f"API Error: {type(e).__name__}: {e}")
    print("Troubleshooting:")
    print("- Verify internet/DNS access to api.groq.com")
    print("- Verify GROQ_API_KEY is active and not revoked")
    print("- Verify the model name is available for your account")
