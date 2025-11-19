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
from browser import get_page_text, get_rendered_html  # CRITICAL: Import browser functions

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
    """
    CRITICAL FIX: Use browser to scrape JavaScript-rendered pages.
    This is essential for web scraping tasks.
    """
    try:
        # First try with browser (for JS-rendered pages)
        content = await get_page_text(url, wait_ms=3000)
        return content
    except Exception as e:
        print(f"Browser scraping failed, trying HTTP: {e}")
        # Fallback to simple HTTP GET
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
    
    # 2. CRITICAL FIX: Extract relative URLs for scraping
    # Look for patterns like: /demo-scrape-data?email=...
    relative_urls = re.findall(r'(/[^\s<>"\'\)]+(?:\?[^\s<>"\'\)]+)?)', text)
    for rel_url in relative_urls:
        rel_url = rel_url.rstrip(".,;:)]")
        # Skip if it's already in submit URLs
        if any(rel_url in s for s in urls["submit"]):
            continue
        
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
        if parsed.path in ["", "/"] or full_url in urls["submit"]: 
            continue

        lower = full_url.lower()
        if any(ext in lower for ext in ['.pdf', '.csv', '.xlsx', '.xls', '.json', '.txt', '.png', '.jpg']):
            if full_url not in urls["download"]: 
                urls["download"].append(full_url)
        elif 'scrape' in lower and full_url not in urls["scrape"]:
            urls["scrape"].append(full_url)
        elif any(k in lower for k in ['api', 'data']):
            if full_url not in urls["api"]: 
                urls["api"].append(full_url)
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
    """Strip markdown code blocks and whitespace."""
    if not isinstance(answer, str): 
        return answer
    answer = re.sub(r'```\w*\s*', '', answer)
    answer = re.sub(r'\s*```', '', answer)
    return answer.strip()


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
                if hasattr(result, 'sum'): 
                    return float(result.sum())
                else: 
                    return float(sum(result))
        
        if hasattr(result, 'item'): 
            return result.item()
        return result
    except Exception as e:
        print(f"Pandas execution failed: {e}")
        try: 
            return float(df.select_dtypes('number').iloc[:, 0].sum())
        except: 
            return str(e)


async def solve_quiz_from_text(quiz_text: str, quiz_url: str) -> Tuple[Any, str, Optional[str]]:
    """Main solver with improved web scraping support."""
    urls = extract_urls_from_text(quiz_text, base_url=quiz_url)
    submit_url = urls["submit"][0] if urls["submit"] else None
    
    print(f"Extracted URLs: {urls}")
    
    try:
        # 1. CRITICAL FIX: Handle Web Scraping Tasks FIRST
        if urls["scrape"]:
            scrape_url = urls["scrape"][0]
            print(f"Web scraping task detected. Scraping: {scrape_url}")
            
            # Use browser to scrape the page
            scraped_content = await scrape_page_content(scrape_url)
            print(f"Scraped content preview: {scraped_content[:300]}")
            
            # Extract the secret/answer from scraped content
            system = """You are extracting specific information from scraped web content.
            Find the secret code, answer, or key information requested in the question.
            Return ONLY the exact value, no explanations, no markdown, no quotes."""
            
            user = f"""Question: {quiz_text}

Scraped Content:
{scraped_content}

Extract the answer:"""
            
            answer = await call_llm(system, user)
            answer = clean_answer(answer)
            print(f"Extracted answer from scrape: {answer}")
            return answer, "string", submit_url
        
        # 2. Handle PDF Downloads
        pdf_url = next((u for u in urls["download"] if '.pdf' in u.lower()), None)
        if pdf_url:
            print(f"PDF task detected: {pdf_url}")
            pdf_bytes = await download_file(pdf_url)
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                # Extract text from all pages
                all_text = ""
                for page in pdf.pages:
                    all_text += page.extract_text() + "\n\n"
                
                system = "Extract the exact answer from this PDF content. Return only the value."
                user = f"Question: {quiz_text}\n\nPDF Text:\n{all_text[:2000]}"
                ans = await call_llm(system, user)
                return clean_answer(ans), "string", submit_url
        
        # 3. Handle Data File Downloads (CSV, Excel, JSON)
        data_url = next((u for u in urls["download"] if u != pdf_url), None)
        if data_url:
            print(f"Data file task detected: {data_url}")
            file_bytes = await download_file(data_url)
            df = parse_data_file(file_bytes, data_url.split('/')[-1])
            print(f"Loaded DataFrame shape: {df.shape}")
            print(f"DataFrame head:\n{df.head()}")
            answer = await solve_data_analysis(quiz_text, df)
            return answer, "number", submit_url

        # 4. Handle API Endpoints (non-scraping)
        elif urls["api"]:
            api_url = urls["api"][0]
            print(f"API task detected: {api_url}")
            
            # Check for custom headers
            headers = {}
            auth_match = re.search(r'Authorization:\s*([^\n]+)', quiz_text, re.IGNORECASE)
            if auth_match: 
                headers['Authorization'] = auth_match.group(1).strip()
            
            data = await fetch_api_data(api_url, headers)
            
            # Convert to string for LLM extraction
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, indent=2)
            else:
                data_str = str(data)
            
            print(f"API response preview: {data_str[:300]}")
            
            system = "Extract ONLY the secret code/answer from this API data. Return strictly the value, no markdown."
            user = f"Question: {quiz_text}\n\nAPI Response:\n{data_str}"
            answer = await call_llm(system, user)
            return clean_answer(answer), "string", submit_url

    except Exception as e:
        print(f"Solver error: {e}")
        import traceback
        traceback.print_exc()

    # 5. Fallback: Use LLM to answer directly
    print("Using fallback LLM answer")
    system = "You are a helpful assistant. Answer the question directly and concisely."
    user = f"Question:\n{quiz_text}\n\nAnswer:"
    answer = await call_llm(system, user)
    return clean_answer(answer), "string", submit_url
