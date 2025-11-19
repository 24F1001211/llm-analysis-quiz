# Complete Testing Guide for LLM Quiz Project

## Pre-requisites Checklist

Before testing, ensure you have:
- [ ] Python 3.8+ installed
- [ ] All project files updated with Gemini code
- [ ] Gemini API key obtained
- [ ] Virtual environment created (recommended)

## Step 1: Environment Setup

### Create Virtual Environment (Recommended)
```bash
# Navigate to your project directory
cd /path/to/your/project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Install Playwright browsers (IMPORTANT!)
playwright install chromium

# Verify installations
pip list | grep -E "fastapi|playwright|google-genai|pandas"
```

## Step 2: Configure Environment Variables

### Create/Update .env file
```bash
# Create .env file in project root
nano .env  # or use any text editor
```

### Add these variables:
```bash
# Student Configuration
STUDENT_SECRET=paneerbuttermasala
STUDENT_EMAIL=24f1001211@ds.study.iitm.ac.in

# Google Gemini API Configuration
GEMINI_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY_HERE

# Quiz Configuration
MAX_QUIZ_SECONDS=170
MAX_RETRIES=2
RETRY_DELAY=1

# Browser Configuration
BROWSER_WAIT_MS=3000
BROWSER_TIMEOUT=60000

# Model Configuration
DEFAULT_MODEL=gemini-2.0-flash-exp
FALLBACK_MODEL=gemini-2.5-flash

# Application Settings
DEBUG_MODE=true

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Logging
LOG_LEVEL=INFO
```

**IMPORTANT**: Replace `YOUR_ACTUAL_GEMINI_API_KEY_HERE` with your real API key!

### Verify .gitignore
```bash
# Check if .env is in .gitignore
cat .gitignore

# If .gitignore doesn't exist or doesn't have .env, add it:
echo ".env" >> .gitignore
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
```

## Step 3: Quick API Test

### Test Gemini Connection
Create a test file `test_gemini.py`:
```python
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key loaded: {api_key[:10]}..." if api_key else "API Key NOT loaded!")

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say 'Hello! API is working!' and nothing else.",
        config=types.GenerateContentConfig(
            temperature=0.1,
        )
    )
    print("✅ Gemini API works!")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")
```

Run it:
```bash
python test_gemini.py
```

Expected output:
```
API Key loaded: AIzaSyBxxx...
✅ Gemini API works!
Response: Hello! API is working!
```

## Step 4: Start the Server

### Option A: Run with Uvicorn directly
```bash
# Start server with auto-reload (for development)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or with debug logging
uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level debug
```

### Option B: Run with Python
```bash
python main.py
```

### Check Server Status
You should see output like:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Open browser and visit: `http://localhost:8000`

Expected response:
```json
{
  "status": "ok",
  "message": "LLM Analysis Quiz Endpoint is running",
  "endpoints": {
    "quiz": "POST /quiz - Submit quiz task",
    "health": "GET /health - Health check"
  }
}
```

## Step 5: Test Endpoints

### Test 1: Health Check
```bash
curl http://localhost:8000/health
```

Expected:
```json
{"status": "healthy", "service": "quiz-endpoint"}
```

### Test 2: Test Endpoint
```bash
curl http://localhost:8000/test
```

Expected:
```json
{
  "status": "ok",
  "message": "Test endpoint working",
  "secret_configured": true,
  "debug_mode": true
}
```

### Test 3: Invalid JSON (should return 400)
```bash
curl -X POST http://localhost:8000/quiz \
  -H "Content-Type: application/json" \
  -d 'invalid json'
```

Expected:
```json
{"detail": "Invalid JSON payload"}
```

### Test 4: Invalid Secret (should return 403)
```bash
curl -X POST http://localhost:8000/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "secret": "wrong_secret",
    "url": "https://example.com"
  }'
```

Expected:
```json
{"detail": "Invalid secret"}
```

### Test 5: Valid Request with Demo
```bash
curl -X POST http://localhost:8000/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "24f1001211@ds.study.iitm.ac.in",
    "secret": "paneerbuttermasala",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }'
```

This will:
1. Accept your request (200 OK)
2. Visit the demo URL
3. Render the JavaScript page
4. Extract the quiz question
5. Use Gemini to solve it
6. Submit the answer
7. Return results

Expected (will take 10-30 seconds):
```json
{
  "status": "ok",
  "message": "Quiz processing completed",
  "quiz_result": {
    "start_url": "https://tds-llm-analysis.s-anand.net/demo",
    "steps": [...],
    "finished": true,
    "reason": "Quiz chain ended"
  }
}
```

## Step 6: Advanced Testing

### Test Browser Functionality
Create `test_browser.py`:
```python
import asyncio
from browser import get_page_text, get_rendered_html

async def test_browser():
    print("Testing browser...")
    try:
        # Test a simple page
        text = await get_page_text("https://example.com")
        print(f"✅ Browser works! Text length: {len(text)}")
        print(f"Preview: {text[:200]}")
    except Exception as e:
        print(f"❌ Browser error: {e}")

asyncio.run(test_browser())
```

Run:
```bash
python test_browser.py
```

### Test Solver Functionality
Create `test_solver.py`:
```python
import asyncio
from solver import call_llm

async def test_solver():
    print("Testing Gemini solver...")
    try:
        system = "You are a helpful assistant."
        user = "What is 2 + 2? Reply with only the number."
        
        answer = await call_llm(system, user)
        print(f"✅ Solver works! Answer: {answer}")
        
        if "4" in answer:
            print("✅ Correct answer!")
        else:
            print(f"⚠️ Unexpected answer: {answer}")
    except Exception as e:
        print(f"❌ Solver error: {e}")

asyncio.run(test_solver())
```

Run:
```bash
python test_solver.py
```

## Step 7: Monitor Logs

### Watch Real-time Logs
In one terminal, keep the server running with debug mode:
```bash
DEBUG_MODE=true python main.py
```

In another terminal, send test requests and watch the logs.

### Key Log Messages to Look For:
- ✅ `Starting quiz for ...` - Quiz started
- ✅ `Quiz completed for ...` - Quiz finished
- ❌ `Error running quiz:` - Something went wrong
- ⚠️ `Invalid secret provided` - Authentication issue

## Step 8: Test with Postman (Alternative)

### Install Postman
Download from: https://www.postman.com/downloads/

### Create Request
1. Method: POST
2. URL: `http://localhost:8000/quiz`
3. Headers:
   - `Content-Type: application/json`
4. Body (raw JSON):
```json
{
  "email": "24f1001211@ds.study.iitm.ac.in",
  "secret": "paneerbuttermasala",
  "url": "https://tds-llm-analysis.s-anand.net/demo"
}
```
5. Click Send

## Troubleshooting Common Issues

### Issue 1: "GEMINI_API_KEY not found"
```bash
# Check if .env file exists
ls -la .env

# Check if it's being loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('GEMINI_API_KEY'))"
```

### Issue 2: "Module not found"
```bash
# Reinstall requirements
pip install -r requirements.txt

# Check if in virtual environment
which python
```

### Issue 3: "Playwright not found"
```bash
# Reinstall Playwright
pip install playwright
playwright install chromium

# Verify installation
playwright --version
```

### Issue 4: Port 8000 already in use
```bash
# Check what's using port 8000
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Use different port
uvicorn main:app --reload --port 8001
```

### Issue 5: Browser timeout
```bash
# Increase timeout in .env
BROWSER_TIMEOUT=120000
BROWSER_WAIT_MS=5000
```

### Issue 6: Gemini API rate limit
```
Error: Resource exhausted (rate limit)
```
Solution: Wait 1 minute or upgrade to paid tier

### Issue 7: "Address already in use"
```bash
# Kill existing server
pkill -f uvicorn
# or find and kill the process
ps aux | grep uvicorn
kill <PID>
```

## Step 9: Load Testing

### Test Multiple Requests
Create `load_test.py`:
```python
import asyncio
import httpx

async def test_request():
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            "http://localhost:8000/quiz",
            json={
                "email": "24f1001211@ds.study.iitm.ac.in",
                "secret": "paneerbuttermasala",
                "url": "https://tds-llm-analysis.s-anand.net/demo"
            }
        )
        return response.status_code

async def load_test(n=5):
    tasks = [test_request() for _ in range(n)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"Results: {results}")

asyncio.run(load_test(3))
```

## Step 10: Pre-Submission Checklist

Before submitting to the evaluation:

- [ ] Server starts without errors
- [ ] Health endpoint returns 200
- [ ] Invalid JSON returns 400
- [ ] Invalid secret returns 403
- [ ] Valid request returns 200
- [ ] Demo endpoint completes successfully
- [ ] Browser renders JavaScript pages
- [ ] Gemini API calls work
- [ ] Logs show quiz steps
- [ ] Response time < 3 minutes
- [ ] .env is in .gitignore
- [ ] GitHub repo is public
- [ ] MIT LICENSE exists
- [ ] README.md exists (optional but good)

## Step 11: Deploy Testing (Optional)

### Test on Cloud (e.g., Railway, Render)
1. Push to GitHub
2. Deploy to cloud service
3. Test with cloud URL:
```bash
curl -X POST https://your-app.railway.app/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "24f1001211@ds.study.iitm.ac.in",
    "secret": "paneerbuttermasala",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }'
```

## Quick Test Script

Save this as `quick_test.sh`:
```bash
#!/bin/bash

echo "🚀 Quick Test Script"
echo "===================="

echo "1. Testing health endpoint..."
curl -s http://localhost:8000/health | python -m json.tool

echo -e "\n2. Testing invalid secret (should be 403)..."
curl -s -X POST http://localhost:8000/quiz \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","secret":"wrong","url":"http://example.com"}' \
  | python -m json.tool

echo -e "\n3. Testing valid request..."
curl -s -X POST http://localhost:8000/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "24f1001211@ds.study.iitm.ac.in",
    "secret": "paneerbuttermasala",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }' | python -m json.tool

echo -e "\n✅ All tests complete!"
```

Make it executable and run:
```bash
chmod +x quick_test.sh
./quick_test.sh
```

## Summary of Testing Flow

1. ✅ Setup environment and install dependencies
2. ✅ Configure .env with Gemini API key
3. ✅ Test Gemini API connection
4. ✅ Start the server
5. ✅ Test health endpoints
6. ✅ Test authentication (400/403 errors)
7. ✅ Test with demo endpoint (full flow)
8. ✅ Monitor logs for issues
9. ✅ Fix any errors
10. ✅ Final checklist before submission

Good luck! 🎉