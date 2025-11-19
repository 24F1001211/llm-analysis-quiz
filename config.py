import os
from dotenv import load_dotenv

load_dotenv()

# Student credentials
STUDENT_SECRET = os.getenv("STUDENT_SECRET", "CHANGE_ME")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL", "you@example.com")

# API keys - Changed from OpenAI to Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate critical settings
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in environment variables")

# Timing constraints
MAX_QUIZ_SECONDS = int(os.getenv("MAX_QUIZ_SECONDS", "170"))  # Under 3 minutes

# Retry settings
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))  # Max retries per question
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "1"))  # Seconds between retries

# Browser settings
BROWSER_WAIT_MS = int(os.getenv("BROWSER_WAIT_MS", "3000"))
BROWSER_TIMEOUT = int(os.getenv("BROWSER_TIMEOUT", "60000"))

# Model settings - Updated for Gemini
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash")

# Logging
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"