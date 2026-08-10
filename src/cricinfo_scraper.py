import asyncio
import json
import pandas as pd
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

async def scrape_match(match_id):
    url = f"https://www.espncricinfo.com/matches/engine/match/{match_id}.html"
    print(f"Navigating to {url}...")
    
    async with async_playwright() as p:
        # Launch Chromium headless
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        await Stealth().apply_stealth_async(page)
        
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            
            # Wait a few seconds for any anti-bot to pass
            await page.wait_for_timeout(3000)
            
            # Look for the __NEXT_DATA__ script tag
            element = page.locator("script#__NEXT_DATA__")
            if await element.count() == 0:
                print("Error: Could not find __NEXT_DATA__ on the page. Dumping HTML for debugging...")
                html = await page.content()
                with open("debug.html", "w", encoding="utf-8") as f:
                    f.write(html)
                return None
                
            json_text = await element.inner_text()
            data = json.loads(json_text)
            
            # Dump to temp file so we can inspect the exact schema
            with open("temp_next_data.json", "w", encoding="utf-8") as f:
                json.dump(data, f)
            print("Successfully dumped __NEXT_DATA__ to temp_next_data.json!")
            
            return data
            
        except Exception as e:
            print(f"Failed to scrape {match_id}: {e}")
            return None
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(scrape_match("1001349"))
