import os
import pandas as pd
from datasets import load_dataset
from src.nlp import CommentaryParser

def analyze_finch():
    print("Loading HuggingFace dataset VinitT/Cricket-Commentary-Sample...")
    # Load dataset
    dataset = load_dataset("VinitT/Cricket-Commentary-Sample", split="train")
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    
    # Filter for Aaron Finch deliveries
    # The input string typically looks like "... batsman name is Aaron Finch ..." or "AJ Finch"
    finch_df = df[df['input'].str.contains('Finch', case=False, na=False)].copy()
    print(f"Found {len(finch_df)} deliveries faced by Aaron Finch.")
    
    if len(finch_df) == 0:
        print("No data found for Aaron Finch.")
        return
        
    # Isolate dismissals vs survivals (the input string has "dismissal is True" or "False")
    # Actually, we can just parse all 293 deliveries to see what lines/lengths he faces most,
    # and what lines/lengths result in a dismissal.
    finch_df['is_dismissal'] = finch_df['input'].str.contains('dismissal is True', case=False, na=False)
    
    # Parse commentary using our new ML models
    parser = CommentaryParser()
    
    lines = []
    lengths = []
    shots = []
    
    print("Running Machine Learning extraction on commentary...")
    for idx, row in finch_df.iterrows():
        commentary = str(row['output']).strip()
        # extract_features returns strings like "Outside Off (98.5%)"
        # We just want the classification part before the parenthesis
        features = parser.extract_features(commentary)
        
        line_clean = features['line'].split(" (")[0]
        length_clean = features['length'].split(" (")[0]
        shot_clean = features['shot'].split(" (")[0]
        
        lines.append(line_clean)
        lengths.append(length_clean)
        shots.append(shot_clean)
        
    finch_df['predicted_line'] = lines
    finch_df['predicted_length'] = lengths
    finch_df['predicted_shot'] = shots
    
    # Save the raw extracted dataset
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "finch_vulnerabilities.csv")
    finch_df.to_csv(out_path, index=False)
    
    print(f"\n--- AARON FINCH VULNERABILITY ANALYSIS ---")
    
    # 1. Total deliveries by Length
    print("\n[Deliveries Faced by Length]")
    print(finch_df['predicted_length'].value_counts())
    
    # 2. Total deliveries by Line
    print("\n[Deliveries Faced by Line]")
    print(finch_df['predicted_line'].value_counts())
    
    # 3. Dismissal Analysis
    dismissals = finch_df[finch_df['is_dismissal'] == True]
    print(f"\n[Total Dismissals Found: {len(dismissals)}]")
    
    if len(dismissals) > 0:
        print("\nDismissals by Length:")
        print(dismissals['predicted_length'].value_counts())
        
        print("\nDismissals by Line:")
        print(dismissals['predicted_line'].value_counts())
        
        print("\nDismissals by Attempted Shot:")
        print(dismissals['predicted_shot'].value_counts())
        
    print(f"\nDetailed CSV saved to {out_path}")

if __name__ == "__main__":
    analyze_finch()
