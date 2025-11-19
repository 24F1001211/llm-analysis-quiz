import sys
import asyncio
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from fastapi.exceptions import RequestValidationError

# Import local config and runner
from config import STUDENT_SECRET, DEBUG_MODE
from quiz_runner import run_quiz

# --- CRITICAL FIX FOR WINDOWS PLAYWRIGHT ---
# This forces the SelectorEventLoop, which is required for Playwright
# to work with asyncio on Windows.
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# -------------------------------------------

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Analysis Quiz Endpoint",
    description="Automated quiz solver for LLM-based data analysis tasks",
    version="2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Convert FastAPI's default 422 into a 400 as per spec."""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid JSON payload", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "LLM Analysis Quiz Endpoint is running",
        "endpoints": {
            "quiz": "POST /quiz - Submit quiz task",
            "health": "GET /health - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "quiz-endpoint"}


@app.post("/")
async def root_post():
    """Handle POST to root - redirect to quiz endpoint."""
    return {"status": "ok", "message": "Use POST /quiz to start the quiz."}


@app.post("/quiz")
async def quiz_endpoint(request: Request):
    """
    Main quiz endpoint that accepts quiz tasks and solves them.
    
    Expected payload:
    {
        "email": "student@example.com",
        "secret": "your-secret",
        "url": "https://quiz-url.com/task-123"
    }
    """
    # Manually parse JSON to detect invalid JSON and return 400
    try:
        json_data = await request.json()
    except Exception as e:
        logger.warning(f"Invalid JSON received: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Validate payload structure
    try:
        payload = QuizRequest(**json_data)
    except ValidationError as e:
        logger.warning(f"Invalid payload structure: {e.errors()}")
        raise HTTPException(
            status_code=400,
            detail={"message": "Invalid payload structure", "errors": e.errors()},
        )

    # Verify secret
    if payload.secret != STUDENT_SECRET:
        logger.warning(f"Invalid secret provided for email: {payload.email}")
        raise HTTPException(status_code=403, detail="Invalid secret")

    logger.info(f"Starting quiz for {payload.email} at {payload.url}")

    # Run the quiz solving process
    try:
        result = await run_quiz(
            email=payload.email,
            secret=payload.secret,
            start_url=payload.url,
        )
        
        logger.info(f"Quiz completed for {payload.email}: {result.get('reason')}")
        
        return {
            "status": "ok",
            "message": "Quiz processing completed",
            "quiz_result": result,
        }
        
    except Exception as e:
        logger.error(f"Error running quiz: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": "Error during quiz execution",
            "error": str(e),
            "quiz_result": {
                "finished": False,
                "reason": f"Exception: {str(e)}"
            }
        }


@app.get("/test")
async def test_endpoint():
    """Test endpoint for debugging."""
    return {
        "status": "ok",
        "message": "Test endpoint working",
        "secret_configured": STUDENT_SECRET != "CHANGE_ME",
        "debug_mode": DEBUG_MODE
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=DEBUG_MODE,
        log_level="debug" if DEBUG_MODE else "info"
    )