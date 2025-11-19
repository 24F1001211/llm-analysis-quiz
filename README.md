# LLM Analysis Quiz Endpoint

An automated quiz solver that handles data-related tasks using headless browsers, data processing tools, and LLM reasoning.

## 🚀 Features

- **Automated Quiz Solving**: Processes quiz chains with multiple questions
- **Multi-format Support**: Handles PDF, CSV, Excel, JSON, images, and more
- **JavaScript Rendering**: Uses Playwright for JS-heavy pages
- **Data Analysis**: Performs filtering, aggregation, statistical analysis
- **LLM-Powered**: Uses GPT-4 for complex reasoning and interpretation
- **Visualization**: Generates charts and graphs when needed
- **Retry Logic**: Intelligent retry mechanism with error feedback
- **Time Management**: Stays within 3-minute time limit

## 📋 Prerequisites

- Python 3.9+
- OpenAI API key
- Internet connection

## 🔧 Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd llm-analysis-quiz
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
playwright install chromium
```

### 4. Configure environment
Create a `.env` file in the project root:

```env
STUDENT_SECRET=your_secret_here
STUDENT_EMAIL=your.email@example.com
OPENAI_API_KEY=sk-your-api-key-here

# Optional settings
MAX_QUIZ_SECONDS=170
MAX_RETRIES=2
DEBUG_MODE=false
DEFAULT_MODEL=gpt-4o
```

## 🏃 Running the Server

### Development mode
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Python directly
```bash
python main.py
```

## 📡 API Endpoints

### POST /quiz
Main endpoint for quiz submission.

**Request:**
```json
{
  "email": "your.email@example.com",
  "secret": "your_secret",
  "url": "https://example.com/quiz-123"
}
```

**Response:**
```json
{
  "status": "ok",
  "message": "Quiz processing completed",
  "quiz_result": {
    "start_url": "https://example.com/quiz-123",
    "finished": true,
    "reason": "Quiz chain completed successfully",
    "total_time": 45.2,
    "success_count": 3,
    "failure_count": 0,
    "steps": [...]
  }
}
```

### GET /
Health check endpoint.

### GET /health
Detailed health status.

## 🧪 Testing

### Test with demo endpoint
```bash
curl -X POST http://localhost:8000/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your.email@example.com",
    "secret": "your_secret",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }'
```

### Test secret validation (should return 403)
```bash
curl -X POST http://localhost:8000/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "secret": "wrong_secret",
    "url": "https://example.com/quiz"
  }'
```

### Test invalid JSON (should return 400)
```bash
curl -X POST http://localhost:8000/quiz \
  -H "Content-Type: application/json" \
  -d 'invalid json'
```

## 🏗️ Architecture

### Core Components

1. **main.py**: FastAPI server with endpoints and error handling
2. **quiz_runner.py**: Orchestrates the quiz-solving process
3. **solver.py**: LLM-powered quiz solver with multi-format support
4. **browser.py**: Headless browser utilities for JS rendering
5. **submitter.py**: Answer submission with validation
6. **config.py**: Configuration and environment variables

### Data Flow

```
Quiz Request → Browser Fetch → Text Extraction → LLM Analysis
                                                        ↓
Answer Submission ← Answer Generation ← Data Processing
                                                        ↓
Next Quiz (if any) ←─────────────────← Response Parsing
```

## 📊 Supported Quiz Types

1. **PDF Analysis**: Extract tables, sum columns, analyze data
2. **CSV/Excel Processing**: Filter, aggregate, transform data
3. **API Interaction**: Fetch and process API responses
4. **Web Scraping**: Extract data from HTML pages
5. **Image Analysis**: OCR and visual processing
6. **Data Visualization**: Generate charts and graphs
7. **Statistical Analysis**: Mean, median, correlations, etc.
8. **Text Processing**: Clean, parse, and analyze text

## 🎯 Design Decisions

### 1. Model Selection
- **Primary**: GPT-4o for complex reasoning and vision tasks
- **Fallback**: GPT-4o-mini for simpler tasks
- **Why**: Balance between capability and cost

### 2. Browser Automation
- **Tool**: Playwright with Chromium
- **Why**: Best JS rendering support, reliable
- **Wait Strategy**: Network idle + 3s buffer for dynamic content

### 3. Retry Logic
- **Max Retries**: 2 per question
- **Strategy**: Re-solve with error feedback
- **Why**: Handles transient failures and improves accuracy

### 4. Time Management
- **Total Limit**: 170 seconds (under 3 min)
- **Per-Step**: Dynamic based on remaining time
- **Why**: Ensures compliance with quiz time limits

### 5. Error Handling
- **Layered**: Try-catch at multiple levels
- **Graceful Degradation**: Fallback to simpler methods
- **Logging**: Comprehensive for debugging

## 🔒 Security Considerations

- Secret validation on all requests
- Input sanitization for URLs and data
- File size limits (1MB for submissions)
- Timeout protection against hanging requests
- No arbitrary code execution

## 🐛 Debugging

Enable debug mode in `.env`:
```env
DEBUG_MODE=true
```

This provides:
- Detailed console output
- Step-by-step progress
- Error stack traces
- Timing information

## 📈 Performance Tips

1. **Use fast models** for simple tasks (gpt-4o-mini)
2. **Cache results** if solving similar questions
3. **Optimize wait times** in browser (reduce if pages load fast)
4. **Parallel processing** for multiple independent tasks
5. **Preload libraries** to reduce startup time

## 🚨 Common Issues

### Issue: Browser not found
**Solution**: Run `playwright install chromium`

### Issue: Timeout errors
**Solution**: Increase `BROWSER_TIMEOUT` in config.py

### Issue: API rate limits
**Solution**: Add delays between requests or use different API key

### Issue: PDF extraction fails
**Solution**: Check if PDF has proper table structure

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is an academic project. Contributions should follow academic integrity guidelines.

## 📞 Support

For issues related to:
- **Quiz endpoint**: Check logs in debug mode
- **API errors**: Verify OpenAI API key and quota
- **Browser issues**: Ensure Playwright is properly installed

## 🎓 Academic Integrity

This code is for educational purposes. When submitting:
1. Understand every component
2. Be able to explain design choices
3. Prepare for viva questions
4. Follow your institution's academic integrity policy

---

**Built with**: FastAPI • Playwright • OpenAI • Pandas • Love for Data ❤️


Migration Guide: OpenAI to Gemini
Overview
This guide helps you migrate your LLM Analysis Quiz project from OpenAI to Google Gemini API.

Why Migrate to Gemini?
Advantages:
Free Tier: Gemini offers generous free tier limits
Multimodal Native: Built from ground up for text, images, audio, video
Latest Models: Access to Gemini 2.0 Flash (reasoning model) and Gemini 2.5 Flash
Long Context: Up to 2 million tokens context window on some models
Cost-Effective: Generally more affordable for production use
Key Differences:
API Structure: Gemini uses system_instruction instead of system role
Client Initialization: Different SDK structure
Model Names: e.g., gemini-2.0-flash-exp vs gpt-4o
Response Format: Slightly different response object structure
Migration Steps
1. Get Gemini API Key
Visit Google AI Studio
Sign in with your Google account
Create an API key
Copy the key for use in your .env file
2. Install New Dependencies
bash
# Uninstall OpenAI
pip uninstall openai

# Install Gemini SDK
pip install google-genai

# Or use the updated requirements.txt
pip install -r requirements.txt
3. Update Environment Variables
Replace in your .env file:

bash
# OLD
OPENAI_API_KEY=sk-proj-...

# NEW
GEMINI_API_KEY=your_gemini_api_key_here
4. Update Configuration Files
Use the updated config.py artifact
Use the updated .env artifact
Use the updated requirements.txt artifact
Use the updated solver.py artifact
5. Update Other Files (if needed)
The following files remain unchanged:

main.py - No changes needed
browser.py - No changes needed
quiz_runner.py - No changes needed
submitter.py - No changes needed (if you have it)
6. Install Playwright (if not done)
bash
playwright install chromium
API Comparison
OpenAI (Old)
python
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.1
)
answer = response.choices[0].message.content.strip()
Gemini (New)
python
from google import genai
from google.genai import types

client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents=user_prompt,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.1,
        max_output_tokens=4096
    )
)
answer = response.text.strip()
Available Gemini Models
Model	Description	Use Case
gemini-2.0-flash-exp	Latest reasoning model (experimental)	Complex problem solving
gemini-2.5-flash	Stable, fast, cost-effective	General purpose, production
gemini-2.5-pro	More capable, longer context	Complex analysis
gemini-1.5-flash	Older but stable	Fallback option
Vision API Changes
OpenAI (Old)
python
messages=[{
    "role": "user",
    "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}}
    ]
}]
Gemini (New)
python
contents=[
    types.Part.from_text(prompt),
    types.Part.from_image_data(
        mime_type="image/jpeg",
        data=base64_image
    )
]
Testing Your Migration
1. Test the API Connection
python
from google import genai

client = genai.Client(api_key="YOUR_API_KEY")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello! Reply with 'API works!'"
)
print(response.text)
2. Test with Demo Endpoint
bash
curl -X POST http://localhost:8000/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@example.com",
    "secret": "paneerbuttermasala",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }'
3. Monitor Logs
bash
# Run with debug mode
DEBUG_MODE=true python main.py
Troubleshooting
Common Issues
1. "GEMINI_API_KEY not found"

bash
# Ensure .env file is in the same directory
# Check the variable name matches exactly
export GEMINI_API_KEY=your_key_here
2. "Rate limit exceeded"

Gemini free tier: 15 RPM (requests per minute)
Add retry logic with exponential backoff
Consider upgrading to paid tier if needed
3. "Model not found"

Check model name spelling
Some models might be region-restricted
Try fallback model: gemini-2.5-flash
4. Vision API errors

Ensure image is base64 encoded
Check mime_type matches actual image format
Maximum image size: 20MB
Cost Comparison
OpenAI GPT-4o
Input: $2.50 / 1M tokens
Output: $10.00 / 1M tokens
Gemini 2.5 Flash (Free Tier)
15 requests per minute
1 million tokens per minute
1,500 requests per day
FREE up to these limits
Gemini 2.5 Flash (Paid)
Input: $0.075 / 1M tokens
Output: $0.30 / 1M tokens
Much cheaper than OpenAI
Project Structure Check
Ensure your project has:

project/
├── .env (updated with GEMINI_API_KEY)
├── .gitignore (must include .env!)
├── config.py (updated)
├── solver.py (updated)
├── requirements.txt (updated)
├── main.py (no changes)
├── browser.py (no changes)
├── quiz_runner.py (no changes)
├── submitter.py (if you have it)
└── README.md (optional)
Alignment with Problem Statement
✅ Checklist
Your project meets the requirements if:

 API endpoint accepts POST with email, secret, url
 Returns 200 for valid requests
 Returns 400 for invalid JSON
 Returns 403 for invalid secrets
 Uses headless browser to render JavaScript pages
 Extracts quiz questions from rendered HTML
 Solves data analysis tasks (CSV, PDF, Excel, JSON)
 Submits answers within 3 minutes
 Follows quiz chains (handles multiple URLs)
 Handles file downloads
 Processes various data formats
 Uses LLM for complex reasoning
 GitHub repo is public with MIT LICENSE
 .env file is in .gitignore (CRITICAL!)
🔧 Recommended Improvements
Add submitter.py if missing:
python
import httpx

async def submit_answer(submit_url, email, secret, quiz_url, answer):
    async with httpx.AsyncClient(timeout=30) as client:
        payload = {
            "email": email,
            "secret": secret,
            "url": quiz_url,
            "answer": answer
        }
        resp = await client.post(submit_url, json=payload)
        data = resp.json()
        return data.get("correct"), data.get("url"), data
Improve error handling in solver.py
Add logging for debugging
Complete visualization functionality
Test with various data types
Final Steps
✅ Update all configuration files
✅ Install new dependencies
✅ Get Gemini API key
✅ Update .env file
✅ Test locally with demo endpoint
✅ Ensure .gitignore includes .env
✅ Push to GitHub (without .env!)
✅ Test the GitHub deployment
✅ Submit to Google Form
Support
If you encounter issues:

Check Gemini API docs: https://ai.google.dev/gemini-api/docs
Review error logs carefully
Test with simple prompts first
Use fallback models if main model fails
Good luck with your project! 🚀

