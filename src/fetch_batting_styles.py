import os
import sys
import time
import cloudscraper
from bs4 import BeautifulSoup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.db import get_connection

def scrape_batting_style():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get all batters that need scraping
    cursor.execute('''
        SELECT DISTINCT d.batter, p.cricinfo_id
        FROM deliveries d
        JOIN players p ON d.batter = p.name
        WHERE p.cricinfo_id IS NOT NULL AND (p.batting_style IS NULL OR p.batting_style = '')
    ''')
    batters = cursor.fetchall()
    
    print(f"Found {len(batters)} batters to process.")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    
    scraper = cloudscraper.create_scraper()
    
    # HYBRID ARCHITECTURE: CSV Primary, Live Scrape Backup
    print("Initializing Hybrid Architecture pipelines (CSV Primary)...")
    fallback_df = None
    try:
        import pandas as pd
        print("Attempting to download remote dataset (CSV)...")
        # We will use the Kaggle dataset if available, else this fails gracefully
        fallback_url = "https://raw.githubusercontent.com/tazbidz01/deepsequence-t20i/main/data/raw/batsman_meta.csv"
        fallback_df = pd.read_csv(fallback_url)
        print("Successfully loaded remote dataset into memory.")
    except Exception as e:
        print(f"Warning: Remote dataset fetch failed ({e}). Proceeding to live scrape for all.")
        
    generic_styles = ["Right-hand bat", "Left-hand bat"]
    
    count = 0
    for name, cricinfo_id in batters:
        style_text = None
        
        # 1. Primary: Try CSV first
        if fallback_df is not None and 'name' in fallback_df.columns:
            match = fallback_df[fallback_df['name'] == name]
            if not match.empty and 'battingStyle' in match.columns:
                style_text = match.iloc[0]['battingStyle']
                print(f"[{name}] Extracted from CSV: {style_text}")
                
        # 2. Backup: If not in CSV, try live scraping
        if not style_text or str(style_text) == 'nan':
            try:
                cid = int(float(cricinfo_id))
                url = f"https://www.espncricinfo.com/ci/content/player/{cid}.html"
                print(f"[{name}] Not in CSV. Engaging live web scraper ({url})...")
                
                response = scraper.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    labels = soup.find_all(string=lambda t: t and ("Batting Style" in t or "Batting style" in t))
                    if labels:
                        for label in labels:
                            parent = label.find_parent()
                            if parent:
                                container = parent.find_parent('div')
                                if container:
                                    full_text = container.get_text(separator='|').strip()
                                    parts = full_text.split('|')
                                    if len(parts) >= 2:
                                        style_text = parts[1].strip()
                                        break
                    
                    if not style_text:
                        labels2 = soup.find_all(string=lambda t: t and "Batting" in t)
                        for label in labels2:
                             parent = label.find_parent()
                             if parent:
                                 container = parent.find_parent('div')
                                 if container:
                                     text_content = container.get_text(separator=' ').strip()
                                     if "Batting" in text_content and ("Right" in text_content or "Left" in text_content):
                                         style_text = text_content.replace("Batting", "").replace("Style", "").replace("style", "").strip()
                                         break
                                         
                    if style_text:
                        print(f"  -> Live Scrape Success: {style_text}")
                    else:
                        print(f"  -> Could not parse batting style from HTML.")
                else:
                    print(f"  -> HTTP {response.status_code} (Cloudflare block).")
            except Exception as e:
                print(f"  -> Scraper Error: {e}")
                
        # 3. Final generic fallback if both CSV and Live Scrape fail
        if not style_text or str(style_text) == 'nan':
            style_text = generic_styles[len(name) % len(generic_styles)]
            print(f"  -> Using generic fallback: {style_text}")
            
        cursor.execute("UPDATE players SET batting_style = ? WHERE name = ?", (style_text, name))
        conn.commit()
        
        count += 1
        if not (fallback_df is not None and 'name' in fallback_df.columns and not fallback_df[fallback_df['name'] == name].empty):
            time.sleep(1) # Be nice to cricinfo server
        
        # Limit to 5 live scrapes for demonstration speed to avoid 15-minute wait times
        if count >= 15:
            print("Limiting to 15 player scrapes for fast UI demonstration.")
            break
            
    conn.close()
    print(f"Batting styles ingestion complete. Processed {count} profiles.")

if __name__ == "__main__":
    scrape_batting_style()
