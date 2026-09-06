import urllib.request
import zipfile
import os
import subprocess

def run_update():
    print('Starting automated dataset update...')
    
    url = 'https://cricsheet.org/downloads/t20s_json.zip'
    raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    dest = os.path.join(raw_dir, 't20s_json.zip')
    
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
        
    print(f'Downloading latest dataset from {url}...')
    urllib.request.urlretrieve(url, dest)
    
    print('Extracting ZIP file...')
    with zipfile.ZipFile(dest, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)
        
    os.remove(dest)
    print('Download and extraction complete.')
    
    print('Running database ingestion...')
    subprocess.run(['python', '-m', 'src.parser'], cwd=os.path.join(os.path.dirname(__file__), '..'))
    print('Update process fully completed.')

if __name__ == '__main__':
    run_update()
