import io
import re
import base64
import json
from typing import Any, Dict, Optional, Tuple, List
from urllib.parse import urljoin, urlparse
from collections import Counter

import httpx
import pandas as pd
import pdfplumber
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

from config import GEMINI_API_KEY, STUDENT_EMAIL
from browser import get_page_text

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


async def call_llm(system_prompt: str, user_prompt: str, image_bytes: Optional[bytes] = None, 
                   audio_bytes: Optional[bytes] = None, model: str = "gemini-2.0-flash-exp") -> str:
    """Call Gemini LLM with system and user prompts, optionally with image or audio."""
    try:
        contents = []
        
        # Add image if provided
        if image_bytes:
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
        
        # Add audio if provided
        if audio_bytes:
            contents.append(types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"))
        
        # Add text prompt
        contents.append(user_prompt)
        
        response = client.models.generate_content(
            model=model,
            contents=contents,
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
                contents=contents if contents else user_prompt,
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
        # Add GitHub token if fetching from GitHub API
        if 'api.github.com' in url:
            if headers is None:
                headers = {}
            # Try to get GitHub token from environment
            import os
            github_token = os.getenv('GITHUB_TOKEN') or os.getenv('GH_TOKEN')
            if github_token:
                headers['Authorization'] = f'token {github_token}'
                print(f"Using GitHub token for authentication")
            else:
                print("Warning: No GitHub token found, may hit rate limits")
        
        resp = await client.get(url, headers=headers or {})
        resp.raise_for_status()
        try:
            return resp.json()
        except json.JSONDecodeError:
            return resp.text


async def scrape_page_content(url: str) -> str:
    try:
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
    
    # Extract Submit URLs
    submission_regions = re.finditer(r"(?:Post|Submit)(?:[\s\S]{0,50}?)to\s+([\s\S]{0,200})", text, re.IGNORECASE)
    for region in submission_regions:
        region_text = region.group(1)
        candidates = re.findall(r'(?:https?://|/)[^\s<>"\'\)]+', region_text)
        for raw_url in candidates:
            full_url = urljoin(base_url, raw_url.rstrip(".,;:)]"))
            parsed = urlparse(full_url)
            if parsed.path not in ["", "/"] and full_url not in urls["submit"]:
                urls["submit"].append(full_url)
    
    # Extract all URLs (absolute and relative)
    all_matches = re.findall(r'(?:https?://|/)([^\s<>"\'\)]+)', text, re.IGNORECASE)
    
    for raw_url in all_matches:
        if not raw_url.startswith(('http://', 'https://', '/')):
            raw_url = '/' + raw_url
        
        full_url = urljoin(base_url, raw_url.rstrip(".,;:)]"))
        parsed = urlparse(full_url)
        
        if parsed.path in ["", "/"] or full_url in urls["submit"]:
            continue

        lower = full_url.lower()
        
        # Categorize URLs
        if any(ext in lower for ext in ['.pdf', '.csv', '.xlsx', '.xls', '.json', '.txt', '.png', '.jpg', '.opus', '.mp3', '.wav']):
            if full_url not in urls["download"]:
                urls["download"].append(full_url)
        elif 'scrape' in lower and full_url not in urls["scrape"]:
            urls["scrape"].append(full_url)
        elif any(k in lower for k in ['api', '/repos/']):
            if full_url not in urls["api"]:
                urls["api"].append(full_url)
            
    return urls


def parse_data_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = filename.lower().split('.')[-1].split('?')[0]  # Handle query params
    
    if ext == 'csv' or ext == 'txt':
        df = pd.read_csv(io.BytesIO(file_bytes))
        try:
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
    if not isinstance(answer, str):
        return answer
    answer = re.sub(r'```\w*\s*', '', answer)
    answer = re.sub(r'\s*```', '', answer)
    return answer.strip().strip('"').strip("'")


async def solve_data_analysis(question: str, df: pd.DataFrame) -> Any:
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    head_str = df.head().to_string()

    system = """You are a Python data analysis expert. Write a SINGLE line of Python pandas code.
    - Variable `df` is loaded.
    - Code must evaluate to result.
    - No print(), no assignment.
    - If question implies "Cutoff"/"Limit", filter values GREATER THAN cutoff.
    - If aggregation implied (sum, count) after filter, do it. Default to SUM for numbers.
    """
    user = f"Question: {question}\n\nInfo:\n{info_str}\n\nHead:\n{head_str}\n\nExpression:"
    
    code_resp = await call_llm(system, user)
    code = clean_answer(code_resp)
    print(f"Generated Pandas Code: {code}")
    
    try:
        result = eval(code, {}, {"df": df, "pd": pd, "np": np})
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


def get_dominant_color(image_bytes: bytes) -> str:
    """Find the most frequent RGB color in an image."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    pixels = list(img.getdata())
    
    # Count colors
    color_counts = Counter(pixels)
    dominant_rgb = color_counts.most_common(1)[0][0]
    
    # Convert to hex
    return f"#{dominant_rgb[0]:02x}{dominant_rgb[1]:02x}{dominant_rgb[2]:02x}"


async def solve_quiz_from_text(quiz_text: str, quiz_url: str) -> Tuple[Any, str, Optional[str]]:
    urls = extract_urls_from_text(quiz_text, base_url=quiz_url)
    submit_url = urls["submit"][0] if urls["submit"] else None
    
    # Default Submit URL for this domain
    if not submit_url and "tds-llm-analysis.s-anand.net" in quiz_url:
        submit_url = "https://tds-llm-analysis.s-anand.net/submit"
        print("Using fallback submit URL")
    
    print(f"Extracted URLs: {urls}")
    
    try:
        # Handle audio transcription (OPUS, MP3, etc.)
        audio_url = next((u for u in urls["download"] if any(ext in u.lower() for ext in ['.opus', '.mp3', '.wav', '.ogg'])), None)
        if audio_url:
            print(f"Audio transcription task detected: {audio_url}")
            audio_bytes = await download_file(audio_url)
            
            system = "Transcribe the audio exactly as spoken. Return words in lowercase with spaces. If there are digits spoken (like 'two one nine'), write them as a single number (219), not spelled out."
            user = f"Question: {quiz_text}\n\nTranscribe this audio file. Format: 'phrase words ### where ### is the 3-digit number:"
            
            answer = await call_llm(system, user, audio_bytes=audio_bytes)
            answer = clean_answer(answer)
            
            # Post-process: convert spelled-out digits to numbers
            # "two one nine" -> "219"
            digit_words = {
                'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
            }
            words = answer.split()
            result = []
            i = 0
            while i < len(words):
                if words[i] in digit_words:
                    # Collect consecutive digit words
                    digits = ''
                    while i < len(words) and words[i] in digit_words:
                        digits += digit_words[words[i]]
                        i += 1
                    result.append(digits)
                else:
                    result.append(words[i])
                    i += 1
            
            answer = ' '.join(result)
            print(f"Processed transcription: {answer}")
            return answer, "string", submit_url
        
        # Handle image analysis (PNG, JPG for color detection)
        image_url = next((u for u in urls["download"] if any(ext in u.lower() for ext in ['.png', '.jpg', '.jpeg'])), None)
        if image_url and 'heatmap' in quiz_text.lower():
            print(f"Image color analysis task detected: {image_url}")
            image_bytes = await download_file(image_url)
            
            # Get dominant color programmatically
            dominant_color = get_dominant_color(image_bytes)
            print(f"Dominant color found: {dominant_color}")
            return dominant_color, "string", submit_url
        
        # Handle web scraping
        if urls["scrape"]:
            scrape_url = urls["scrape"][0]
            print(f"Web scraping task detected: {scrape_url}")
            scraped_content = await scrape_page_content(scrape_url)
            
            system = "Extract the answer/code from scraped content. Return ONLY the value."
            user = f"Question: {quiz_text}\n\nMy email is: {STUDENT_EMAIL}\n\nScraped Content:\n{scraped_content}\n\nAnswer:"
            
            answer = await call_llm(system, user)
            return clean_answer(answer), "string", submit_url
        
        # Handle PDF files
        pdf_url = next((u for u in urls["download"] if '.pdf' in u.lower()), None)
        if pdf_url:
            print(f"PDF processing task detected: {pdf_url}")
            pdf_bytes = await download_file(pdf_url)
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                all_text = "\n".join([p.extract_text() for p in pdf.pages])
                
            system = "Extract the exact answer. Return only the value."
            user = f"Question: {quiz_text}\n\nMy email is: {STUDENT_EMAIL}\n\nPDF Text:\n{all_text[:2000]}"
            ans = await call_llm(system, user)
            return clean_answer(ans), "string", submit_url
        
        # Handle CSV/data files
        data_url = next((u for u in urls["download"] if any(ext in u.lower() for ext in ['.csv', '.xlsx', '.xls'])), None)
        if data_url:
            print(f"Data analysis task detected: {data_url}")
            file_bytes = await download_file(data_url)
            df = parse_data_file(file_bytes, data_url.split('/')[-1])
            
            # Check if this is a normalization task
            if 'normalize' in quiz_text.lower() or 'snake_case' in quiz_text.lower():
                print("CSV normalization task detected")
                print(f"Original DataFrame:\n{df}")
                
                # Strip whitespace from all string columns
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.strip()
                
                # Normalize column names to snake_case
                df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
                
                # Parse and normalize dates with multiple format support
                for col in df.columns:
                    if 'date' in col or 'joined' in col:
                        # Parse with explicit format detection, prioritizing US format (MM/DD/YY)
                        parsed_dates = []
                        for date_str in df[col]:
                            if pd.isna(date_str):
                                parsed_dates.append(None)
                                continue
                            date_str = str(date_str).strip()
                            # Try US format first (MM/DD/YY or MM/DD/YYYY)
                            try:
                                dt = pd.to_datetime(date_str, format='%m/%d/%y')
                                parsed_dates.append(dt.strftime('%Y-%m-%d'))
                            except:
                                try:
                                    dt = pd.to_datetime(date_str, format='%Y-%m-%d')
                                    parsed_dates.append(dt.strftime('%Y-%m-%d'))
                                except:
                                    try:
                                        # Handle "1 Feb 2024" format
                                        dt = pd.to_datetime(date_str, format='%d %b %Y')
                                        parsed_dates.append(dt.strftime('%Y-%m-%d'))
                                    except:
                                        # Fallback to pandas parser
                                        dt = pd.to_datetime(date_str, dayfirst=False)
                                        parsed_dates.append(dt.strftime('%Y-%m-%d'))
                        df[col] = parsed_dates
                
                # Ensure numeric columns are integers
                for col in df.columns:
                    if 'value' in col or 'id' in col:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                
                # Sort by id if present
                if 'id' in df.columns:
                    df = df.sort_values('id')
                
                print(f"Normalized DataFrame:\n{df}")
                
                # Convert to JSON
                result = df.to_json(orient='records')
                print(f"JSON result: {result}")
                return result, "string", submit_url
            else:
                answer = await solve_data_analysis(quiz_text, df)
                return answer, "number", submit_url
        
        # Handle JSON files with GitHub API tasks
        json_url = next((u for u in urls["download"] if '.json' in u.lower()), None)
        if json_url:
            print(f"JSON file detected: {json_url}")
            file_bytes = await download_file(json_url)
            data = json.loads(file_bytes.decode('utf-8'))

            # --- NEW SENTIMENT HANDLER ---
            if 'sentiment' in quiz_text.lower() and isinstance(data, list):
                print("Sentiment analysis task detected")
                positive_count = 0
                for item in data:
                    label = str(item.get('sentiment', '') or item.get('label', '')).lower()
                    if 'positive' in label:
                        positive_count += 1
                print(f"Counted {positive_count} positive tweets.")
                return positive_count, "number", submit_url
            # -----------------------------
            
            # Check if this is a GitHub tree API task
            if 'owner' in data and 'repo' in data and 'sha' in data:
                print("GitHub API task detected")
                owner = data['owner']
                repo = data['repo']
                sha = data['sha']
                path_prefix = data.get('pathPrefix', '')
                extension = data.get('extension', '.md')
                
                print(f"GitHub params: owner={owner}, repo={repo}, sha={sha}, prefix={path_prefix}, ext={extension}")
                
                # Fetch GitHub tree with error handling for rate limits
                github_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"
                try:
                    tree_data = await fetch_api_data(github_url)
                except Exception as e:
                    if '403' in str(e) or 'rate limit' in str(e).lower():
                        print("GitHub rate limit hit, waiting and retrying...")
                        import asyncio
                        await asyncio.sleep(5)
                        try:
                            tree_data = await fetch_api_data(github_url)
                        except:
                            # If still fails, try to compute from expected pattern
                            print("Still rate limited, cannot fetch tree")
                            raise
                    else:
                        raise
                
                # Count files matching criteria
                count = 0
                for item in tree_data.get('tree', []):
                    path = item.get('path', '')
                    if path.startswith(path_prefix) and path.endswith(extension):
                        count += 1
                        print(f"  Matched: {path}")
                
                # Add offset based on email length
                offset = len(STUDENT_EMAIL) % 2
                final_answer = count + offset
                
                print(f"Count: {count}, Email length: {len(STUDENT_EMAIL)}, Offset: {offset}, Final: {final_answer}")
                return final_answer, "number", submit_url
        
        # Handle API endpoints
        if urls["api"]:
            api_url = urls["api"][0]
            print(f"API task detected: {api_url}")
            headers = {}
            auth_match = re.search(r'Authorization:\s*([^\n]+)', quiz_text, re.IGNORECASE)
            if auth_match:
                headers['Authorization'] = auth_match.group(1).strip()
            
            data = await fetch_api_data(api_url, headers)
            data_str = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
            
            system = "Extract ONLY the answer from this data. Return strictly the value."
            user = f"Question: {quiz_text}\n\nMy email is: {STUDENT_EMAIL}\n\nData:\n{data_str}"
            answer = await call_llm(system, user)
            return clean_answer(answer), "string", submit_url

    except Exception as e:
        print(f"Solver error: {e}")
        import traceback
        traceback.print_exc()

    # Fallback to LLM
    print("Using fallback LLM answer")
    # UPDATED STRICT PROMPT
    system = """You are a data extraction engine.
    - Return ONLY the raw answer value.
    - Do NOT return JSON format.
    - Do NOT return XML.
    - Do NOT explain.
    - If the answer is a URL, return just the URL.
    - If the answer is a number, return just the number.
    """
    user = f"Question:\n{quiz_text}\n\nMy email is: {STUDENT_EMAIL}\n\nAnswer:"
    answer = await call_llm(system, user)
    return clean_answer(answer), "string", submit_url
