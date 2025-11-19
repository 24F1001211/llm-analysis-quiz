from contextlib import asynccontextmanager
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup


@asynccontextmanager
async def browser_context():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        try:
            yield context
        finally:
            await context.close()
            await browser.close()


async def get_rendered_html(url: str, wait_ms: int = 2000) -> str:
    """
    Open the URL in a headless browser, wait for JS to execute,
    and return the rendered HTML.
    """
    async with browser_context() as context:
        page = await context.new_page()
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_timeout(wait_ms)
        content = await page.content()
        return content


async def get_page_text(url: str, wait_ms: int = 2000) -> str:
    """
    Extract visible text AND preserve link URLs so the LLM can see them.
    """
    async with browser_context() as context:
        page = await context.new_page()
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_timeout(wait_ms)
        
        # Get full HTML content
        content = await page.content()
        
        # Parse with BeautifulSoup to handle links smarter
        soup = BeautifulSoup(content, "html.parser")
        
        # Append (href) to every link text so Regex can find it
        # e.g. <a href="data.csv">Download</a> -> "Download (data.csv)"
        for a in soup.find_all('a', href=True):
            url = a['href']
            text = a.get_text(strip=True)
            if url and text:
                a.replace_with(f"{text} ({url})")
        
        # Return the modified text
        return soup.get_text(separator='\n', strip=True)