import urllib.request
import zipfile
import io
import os

url = "https://cricsheet.org/downloads/t20s_male_json.zip"
print(f"Downloading T20s JSON zip from {url}...")

# Add User-Agent header as Cricsheet might block default urllib agents
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        zip_data = response.read()
        
    print("Download finished. Extracting 5 sample matches...")
    
    target_dir = os.path.join("data", "raw")
    os.makedirs(target_dir, exist_ok=True)
    
    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        # Filter only json files, excluding README etc.
        json_files = [f for f in z.namelist() if f.endswith('.json')]
        
        # Extract the first 5 JSON files
        for file in json_files[:5]:
            z.extract(file, target_dir)
            print(f"Extracted {file}")
            
    print(f"\nSuccessfully populated '{target_dir}' with sample JSON files!")
    
except Exception as e:
    print(f"Error during download/extraction: {e}")
