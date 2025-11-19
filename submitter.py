from typing import Any, Dict, Optional, Tuple

import httpx


async def submit_answer(
    submit_url: str,
    email: str,
    secret: str,
    quiz_url: str,
    answer: Any,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Send answer to submit_url as per spec.
    Returns (correct, next_url, full_response_json).
    """
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(submit_url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    correct = bool(data.get("correct", False))
    next_url = data.get("url")
    return correct, next_url, data
