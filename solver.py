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

from config import GEMINI_API_KEY

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
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers or {})
        resp.raise_for_status()
        try:
            return resp.json()
        except json.JSONDecodeError:
            return resp.text


def extract_urls_from_text(text: str, base_url: str = "") -> Dict[str, List[str]]:
    """Robust URL extractor."""
    urls = {"submit": [], "download": [], "api": [], "general": []}
    
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
    
    # 2. Extract Download/API URLs
    all_matches = re.findall(r'(?:https?://|/|(?<=\())([\w\-.%]+\.(?:csv|pdf|xlsx?|json|txt))|((?:https?://|/)[^\s<>"\'\)]+)', text, re.IGNORECASE)
    potential_urls = [m[0] if m[0] else m[1] for m in all_matches]

    for raw_url in potential_urls:
        full_url = urljoin(base_url, raw_url.rstrip(".,;:)]"))
        parsed = urlparse(full_url)
        if parsed.path in ["", "/"] or full_url in urls["submit"]: continue

        lower = full_url.lower()
        if any(ext in lower for ext in ['.pdf', '.csv', '.xlsx', '.xls', '.json', '.txt', '.png', '.jpg']):
            if full_url not in urls["download"]: urls["download"].append(full_url)
        elif any(k in lower for k in ['api', 'scrape', 'data']):
            if full_url not in urls["api"]: urls["api"].append(full_url)
        elif 'submit' in lower and full_url not in urls["submit"]:
             urls["submit"].append(full_url)
            
    return urls


def parse_data_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parses data files with smart header detection."""
    ext = filename.lower().split('.')[-1]
    
    if ext == 'csv' or ext == 'txt':
        # Try reading normally first
        df = pd.read_csv(io.BytesIO(file_bytes))
        
        # --- CRITICAL FIX: HEADER DETECTION ---
        # If the column names look like numbers (e.g., '96903'), assume header=None
        # This fixes the "missing first row" bug in summation tasks.
        try:
            # Check if all column names are numbers
            if all(col.isdigit() for col in df.columns.astype(str)):
                print("Detected numeric header, reloading with header=None")
                df = pd.read_csv(io.BytesIO(file_bytes), header=None)
        except:
            pass
        return df
        
    elif ext in ['xlsx', 'xls']:
        return pd.read_excel(io.BytesIO(file_bytes))
    elif ext == 'json':
        data = json.loads(file_bytes.decode('utf-8'))
        return pd.DataFrame(data if isinstance(data, list) else [data])
    
    raise ValueError(f"Unsupported file format: {ext}")


def clean_answer(answer: str) -> str:
    """Strip markdown code blocks, whitespace, and quotes."""
    if not isinstance(answer, str): return answer
    # Remove markdown code blocks
    answer = re.sub(r'```\w*\s*', '', answer)
    answer = re.sub(r'\s*```', '', answer)
    # Remove surrounding whitespace and quotes (CRITICAL FIX)
    return answer.strip().strip('"').strip("'")


async def solve_data_analysis(question: str, df: pd.DataFrame) -> Any:
    """Generate and execute Pandas code to answer the question."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    head_str = df.head().to_string()

    system = """You are a Python data analysis expert.
    Write a SINGLE line of Python pandas code to answer the user's question.
    - The variable `df` is already loaded.
    - The code must evaluate to the final result.
    - Do NOT use print(). Do NOT assign variables.
    - If the question mentions "Cutoff" or "Limit", filter values GREATER THAN the cutoff.
    - If the question implies aggregation (sum, average) after filtering, assume SUM unless specified.
    - Example: df[df[0] > 15000][0].sum()
    """
    user = f"Question: {question}\n\nInfo:\n{info_str}\n\nHead:\n{head_str}\n\nExpression:"
    
    code_resp = await call_llm(system, user)
    code = clean_answer(code_resp)
    print(f"Generated Pandas Code: {code}")
    
    try:
        result = eval(code, {}, {"df": df, "pd": pd, "np": np})
        
        # Auto-sum list results
        if isinstance(result, (list, pd.Series, np.ndarray)):
            if hasattr(result, '__len__') and len(result) > 1:
                 print("Detected list result, auto-summing...")
                 if hasattr(result, 'sum'): return float(result.sum())
                 else: return float(sum(result))
        
        if hasattr(result, 'item'): return result.item()
        return result
    except Exception as e:
        print(f"Pandas failed: {e}")
        try: return float(df.select_dtypes('number').iloc[:, 0].sum())
        except: return str(e)


async def solve_quiz_from_text(quiz_text: str, quiz_url: str) -> Tuple[Any, str, Optional[str]]:
    urls = extract_urls_from_text(quiz_text, base_url=quiz_url)
    submit_url = urls["submit"][0] if urls["submit"] else None
    
    try:
        # 1. Handle File Downloads
        pdf_url = next((u for u in urls["download"] if '.pdf' in u.lower()), None)
        data_url = next((u for u in urls["download"] if u != pdf_url), None)

        if pdf_url:
            pdf_bytes = await download_file(pdf_url)
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text_content = pdf.pages[0].extract_text()
                ans = await call_llm("Extract the exact answer value.", f"Q: {quiz_text}\nText: {text_content}")
                return clean_answer(ans), "string", submit_url
        
        elif data_url:
            file_bytes = await download_file(data_url)
            df = parse_data_file(file_bytes, data_url.split('/')[-1])
            answer = await solve_data_analysis(quiz_text, df)
            return answer, "number", submit_url

        # 2. Handle API / Scrape
        elif urls["api"]:
            api_url = urls["api"][0]
            headers = {}
            auth_match = re.search(r'Authorization:\s*([^\n]+)', quiz_text, re.IGNORECASE)
            if auth_match: headers['Authorization'] = auth_match.group(1).strip()
            
            data = await fetch_api_data(api_url, headers)
            
            # --- CRITICAL FIX: Treat JSON API data as text for extraction ---
            # This handles { "secret": "CODE" } much better than converting to DataFrame
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data)
            else:
                data_str = str(data)
                
            system = "Extract ONLY the secret code/answer from this data. Return strictly the value, no markdown."
            user = f"Question: {quiz_text}\n\nData:\n{data_str}"
            answer = await call_llm(system, user)
            return clean_answer(answer), "string", submit_url

    except Exception as e:
        print(f"Solver error: {e}")

    # 3. Fallback
    system = "You are a helpful assistant. Answer the question directly."
    user = f"Question:\n{quiz_text}\n\nAnswer:"
    answer = await call_llm(system, user)
    return clean_answer(answer), "string", submit_url
