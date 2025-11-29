import io
import re
import base64
import json
from typing import Any, Dict, Optional, Tuple, List
from urllib.parse import urljoin, urlparse

import httpx
import pandas as pd
import pdfplumber
import numpy as np
from google import genai
from google.genai import types

from config import GEMINI_API_KEY, STUDENT_EMAIL
from browser import get_page_text

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


async def call_llm(system_prompt: str, user_prompt: str, model: str = "gemini-2.0-flash-exp") -> str:
    """Call Gemini LLM with system and user prompts."""
    try:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                max_output_tokens=4096,
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                )
            )
            return response.text.strip()
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            raise


async def download_file(url: str, headers: Optional[Dict] = None) -> bytes:
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers or {})
        resp.raise_for_status()
        return resp.content


async def fetch_api_data(url: str, headers: Optional[Dict] = None) -> Any:
    """Fetch data from API - returns JSON or text."""
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers or {})
        resp.raise_for_status()
        try:
            return resp.json()
        except json.JSONDecodeError:
            return resp.text


async def scrape_page_content(url: str) -> str:
    try:
        # First try with browser (for JS-rendered pages)
        content = await get_page_text(url, wait_ms=3000)
        return content
    except Exception as e:
        print(f"Browser scraping failed, trying HTTP: {e}")
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.text
        except Exception as e2:
            print(f"HTTP scraping also failed: {e2}")
            return ""


def extract_urls_from_text(text: str, base_url: str = "") -> Dict[str, List[str]]:
    """Robust URL extractor with relative URL support."""
    urls = {"submit": [], "download": [], "scrape": [], "api": [], "general": []}
    
    # 1. Extract Submit URLs
    submission_regions = re.finditer(r"(?:Post|Submit)(?:[\s\S]{0,50}?)to\s+([\s\S]{0,200})", text, re.IGNORECASE)
    for region in submission_regions:
        region_text = region.group(1)
        candidates = re.findall(r'(?:https?://|/)[^\s<>"\'\)]+', region_text)
        for raw_url in candidates:
            full_url = urljoin(base_url, raw_url.rstrip(".,;:)]"))
            parsed = urlparse(full_url)
            if parsed.path not in ["", "/"] and full_url not in urls["submit"]:
                urls["submit"].append(full_url)
    
    # 2. Extract relative URLs for scraping
    relative_urls = re.findall(r'(/[^\s<>"\'\)]+(?:\?[^\s<>"\'\)]+)?)', text)
    for rel_url in relative_urls:
        rel_url = rel_url.rstrip(".,;:)]")
        if any(rel_url in s for s in urls["submit"]): continue
        
        full_url = urljoin(base_url, rel_url)
        lower = full_url.lower()
        
        if 'scrape' in lower and full_url not in urls["scrape"]:
            urls["scrape"].append(full_url)
        elif any(k in lower for k in ['api', 'data']) and full_url not in urls["api"]:
            urls["api"].append(full_url)
    
    # 3. Extract absolute URLs
    all_matches = re.findall(r'(?:https?://|/|(?<=\())([\w\-.%]+\.(?:csv|pdf|xlsx?|json|txt))|((?:https?://|/)[^\s<>"\'\)]+)', text, re.IGNORECASE)
    potential_urls = [m[0] if m[0] else m[1] for m in all_matches]

    for raw_url in potential_urls:
        full_url = urljoin(base_url, raw_url.rstrip(".,;:)]"))
        parsed = urlparse(full_url)
        if parsed.path in ["", "/"] or full_url in urls["submit"]: continue

        lower = full_url.lower()
        if any(ext in lower for ext in ['.pdf', '.csv', '.xlsx', '.xls', '.json', '.txt', '.png', '.jpg']):
            if full_url not in urls["download"]: urls["download"].append(full_url)
        elif 'scrape' in lower and full_url not in urls["scrape"]:
            urls["scrape"].append(full_url)
        elif any(k in lower for k in ['api', 'data']):
            if full_url not in urls["api"]: urls["api"].append(full_url)
        elif 'submit' in lower and full_url not in urls["submit"]:
            urls["submit"].append(full_url)
            
    return urls


def parse_data_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = filename.lower().split('.')[-1]
    if ext == 'csv' or ext == 'txt':
        df = pd.read_csv(io.BytesIO(file_bytes))
        try:
            if all(col.isdigit() for col in df.columns.astype(str)):
                print("Detected numeric header, reloading with header=None")
                df = pd.read_csv(io.BytesIO(file_bytes), header=None)
        except: pass
        return df
    elif ext in ['xlsx', 'xls']:
        return pd.read_excel(io.BytesIO(file_bytes))
    elif ext == 'json':
        data = json.loads(file_bytes.decode('utf-8'))
        return pd.DataFrame(data if isinstance(data, list) else [data])
    raise ValueError(f"Unsupported file format: {ext}")


def clean_answer(answer: str) -> str:
    if not isinstance(answer, str): return answer
    answer = re.sub(r'```\w*\s*', '', answer)
    answer = re.sub(r'\s*```', '', answer)
    return answer.strip().strip('"').strip("'")


async def solve_data_analysis(question: str, df: pd.DataFrame) -> Any:
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    head_str = df.head().to_string()

    # --- UPDATED PROMPT FOR DATA CLEANING ---
    system = """You are a Python data analysis expert. Write a SINGLE line of Python pandas code.
    - Variable `df` is loaded.
    - Code must evaluate to the result. No print().
    - If "Normalize" or "JSON" is asked:
      1. Rename columns to snake_case (e.g. 'Joined' -> 'joined').
      2. Convert dates to ISO-8601 YYYY-MM-DD: `pd.to_datetime(df['joined']).dt.strftime('%Y-%m-%d')`.
      3. Return JSON with `df.to_json(orient='records')`.
    - If "Cutoff"/"Limit": filter values GREATER THAN cutoff.
    - If aggregation implied: default to SUM.
    """
    user = f"Question: {question}\n\nInfo:\n{info_str}\n\nHead:\n{head_str}\n\nExpression:"
    
    code_resp = await call_llm(system, user)
    code = clean_answer(code_resp)
    print(f"Generated Pandas Code: {code}")
    
    try:
        result = eval(code, {}, {"df": df, "pd": pd, "np": np})
        if isinstance(result, (list, pd.Series, np.ndarray)):
            if hasattr(result, '__len__') and len(result) > 1:
                 # Don't sum if it's a JSON string output
                 if isinstance(result[0], str) and '{' in str(result[0]):
                     return result
                 print("Detected list result, auto-summing...")
                 if hasattr(result, 'sum'): return float(result.sum())
                 else: return float(sum(result))
        if hasattr(result, 'item'): return result.item()
        return result
    except Exception as e:
        print(f"Pandas execution failed: {e}")
        try: return float(df.select_dtypes('number').iloc[:, 0].sum())
        except: return str(e)


async def handle_github_task(text: str, json_data: Dict) -> Any:
    """Special handler for GitHub API chaining tasks."""
    try:
        # Extract params from the loaded JSON (which came from gh-tree.json)
        # Usually it's a list with one dict, or just a dict
        if isinstance(json_data, list): params = json_data[0]
        else: params = json_data
        
        owner = params.get('owner')
        repo = params.get('repo')
        sha = params.get('sha') or 'main'
        path_prefix = params.get('pathPrefix', '')
        
        # Construct Real GitHub API URL
        tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"
        print(f"Fetching GitHub Tree: {tree_url}")
        
        # Fetch the tree
        async with httpx.AsyncClient() as client:
            resp = await client.get(tree_url)
            tree_data = resp.json()
            
        # Count files matching logic
        files = tree_data.get('tree', [])
        count = 0
        for f in files:
            path = f.get('path', '')
            if path.startswith(path_prefix) and path.endswith('.md'):
                count += 1
                
        # Personalization (email length offset)
        offset = len(STUDENT_EMAIL) % 2
        return count + offset
        
    except Exception as e:
        print(f"GitHub Handler failed: {e}")
        return 0


async def solve_quiz_from_text(quiz_text: str, quiz_url: str) -> Tuple[Any, str, Optional[str]]:
    urls = extract_urls_from_text(quiz_text, base_url=quiz_url)
    submit_url = urls["submit"][0] if urls["submit"] else None
    
    if not submit_url and "tds-llm-analysis.s-anand.net" in quiz_url:
        submit_url = "https://tds-llm-analysis.s-anand.net/submit"
        print("Using fallback submit URL")
    
    try:
        if urls["scrape"]:
            scrape_url = urls["scrape"][0]
            print(f"Web scraping task detected: {scrape_url}")
            scraped_content = await scrape_page_content(scrape_url)
            system = "Extract the answer/code. Return ONLY the value."
            user = f"Question: {quiz_text}\n\nEmail: {STUDENT_EMAIL}\n\nContent:\n{scraped_content}"
            answer = await call_llm(system, user)
            return clean_answer(answer), "string", submit_url
        
        pdf_url = next((u for u in urls["download"] if '.pdf' in u.lower()), None)
        if pdf_url:
            pdf_bytes = await download_file(pdf_url)
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                all_text = "\n".join([p.extract_text() for p in pdf.pages])
                system = "Extract the exact answer. Return only the value."
                user = f"Question: {quiz_text}\n\nEmail: {STUDENT_EMAIL}\n\nPDF:\n{all_text[:2000]}"
                ans = await call_llm(system, user)
                return clean_answer(ans), "string", submit_url
        
        data_url = next((u for u in urls["download"] if u != pdf_url), None)
        if data_url:
            file_bytes = await download_file(data_url)
            
            # --- FIX: DETECT GITHUB TASK ---
            if 'gh-tree.json' in data_url or 'github' in quiz_text.lower():
                print("Detected GitHub Tree task")
                json_data = json.loads(file_bytes.decode('utf-8'))
                answer = await handle_github_task(quiz_text, json_data)
                return answer, "number", submit_url
                
            df = parse_data_file(file_bytes, data_url.split('/')[-1])
            answer = await solve_data_analysis(quiz_text, df)
            return answer, "number", submit_url

        elif urls["api"]:
            api_url = urls["api"][0]
            headers = {}
            auth_match = re.search(r'Authorization:\s*([^\n]+)', quiz_text, re.IGNORECASE)
            if auth_match: headers['Authorization'] = auth_match.group(1).strip()
            
            data = await fetch_api_data(api_url, headers)
            data_str = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
            
            system = "Extract ONLY the answer. Return strictly the value."
            user = f"Question: {quiz_text}\n\nEmail: {STUDENT_EMAIL}\n\nData:\n{data_str}"
            answer = await call_llm(system, user)
            return clean_answer(answer), "string", submit_url

    except Exception as e:
        print(f"Solver error: {e}")

    # Fallback
    system = "You are a helpful assistant. Replace placeholders like <your email> with the actual email."
    user = f"Question:\n{quiz_text}\n\nMy email is: {STUDENT_EMAIL}\n\nAnswer:"
    answer = await call_llm(system, user)
    return clean_answer(answer), "string", submit_url
