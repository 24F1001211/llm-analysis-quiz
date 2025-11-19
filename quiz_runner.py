import time
from typing import Any, Dict

from config import MAX_QUIZ_SECONDS, STUDENT_EMAIL
from browser import get_page_text
from solver import solve_quiz_from_text
from submitter import submit_answer


async def run_quiz(email: str, secret: str, start_url: str) -> Dict[str, Any]:
    """
    Run the quiz chain starting from start_url, staying within MAX_QUIZ_SECONDS.
    Returns a summary dict for logging / debugging.
    """
    start_time = time.time()
    current_url = start_url
    summary = {
        "start_url": start_url,
        "steps": [],
        "finished": False,
        "reason": None,
    }

    # For safety, limit number of hops
    max_steps = 10

    for step in range(max_steps):
        if current_url is None:
            summary["finished"] = True
            summary["reason"] = "No further URL provided"
            break

        if time.time() - start_time > MAX_QUIZ_SECONDS:
            summary["finished"] = False
            summary["reason"] = "Time limit exceeded"
            break

        step_info: Dict[str, Any] = {"quiz_url": current_url}

        # 1) Load quiz page and get visible text
        quiz_text = await get_page_text(current_url)
        step_info["quiz_text_preview"] = quiz_text[:500]

        # 2) Solve quiz (compute answer + find submit URL)
        answer, answer_type, submit_url = await solve_quiz_from_text(
            quiz_text, current_url
        )

        step_info["answer"] = answer
        step_info["answer_type"] = answer_type
        step_info["submit_url"] = submit_url

        if not submit_url:
            # If no submit URL is found, we can't continue.
            summary["steps"].append(step_info)
            summary["finished"] = False
            summary["reason"] = "No submit URL found"
            break

        # 3) Submit answer
        correct, next_url, resp_json = await submit_answer(
            submit_url=submit_url,
            email=email or STUDENT_EMAIL,
            secret=secret,
            quiz_url=current_url,
            answer=answer,
        )

        step_info["submission_correct"] = correct
        step_info["submission_response"] = resp_json
        step_info["next_url"] = next_url

        summary["steps"].append(step_info)

        if not next_url:
            # Quiz ended
            summary["finished"] = True
            summary["reason"] = "Quiz chain ended"
            break

        # Otherwise continue with the next URL
        current_url = next_url

    else:
        summary["finished"] = False
        summary["reason"] = "Max steps reached"

    return summary
