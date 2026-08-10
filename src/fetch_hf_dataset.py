import os
import pandas as pd
from datasets import load_dataset
from src.nlp import CommentaryParser

def bootstrap_dataset():
    print("Downloading HuggingFace dataset VinitT/Cricket-Commentary-Sample...")
    dataset = load_dataset("VinitT/Cricket-Commentary-Sample", split="train")
    
    # We only care about the 'output' column, which contains the raw commentary text
    df = pd.DataFrame(dataset)
    texts = df['output'].tolist()
    
    print(f"Total rows downloaded: {len(texts)}")
    
    parser = CommentaryParser()
    
    labeled_data = []
    
    print("Applying Weak Supervision via existing Regex engine...")
    for text in texts:
        # Clean the text (remove newlines, extra spaces)
        clean_text = str(text).strip()
        if not clean_text:
            continue
            
        features = parser.extract_features(clean_text)
        
        # We want rows that have AT LEAST ONE non-Unknown feature so the ML model can learn something.
        # But to be very strict and train a highly accurate model, let's keep rows that have 
        # at least a known length OR a known line OR a known shot.
        if features['line'] != 'Unknown' or features['length'] != 'Unknown' or features['shot'] != 'Unknown':
            labeled_data.append({
                'text': clean_text,
                'line': features['line'],
                'length': features['length'],
                'shot': features['shot']
            })
            
    df_labeled = pd.DataFrame(labeled_data)
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hf_commentary_labels.csv")
    
    df_labeled.to_csv(out_path, index=False)
    print(f"Successfully bootstrapped {len(df_labeled)} labeled rows!")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    bootstrap_dataset()
