import os
import joblib
import re

class CommentaryParser:
    def __init__(self):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        
        # Load the TF-IDF Vectorizer
        self.vectorizer = joblib.load(os.path.join(models_dir, "nlp_vectorizer.pkl"))
        
        # Load the ML Classifiers
        self.model_line = joblib.load(os.path.join(models_dir, "nlp_model_line.pkl"))
        self.model_length = joblib.load(os.path.join(models_dir, "nlp_model_length.pkl"))
        self.model_shot = joblib.load(os.path.join(models_dir, "nlp_model_shot.pkl"))

    def extract_features(self, text):
        """
        Takes raw commentary text and returns ML-predicted Line, Length, and Shot
        along with their confidence scores.
        """
        # Vectorize the text
        X = self.vectorizer.transform([text])
        
        # Predict Line
        line_pred = self.model_line.predict(X)[0]
        line_conf = max(self.model_line.predict_proba(X)[0])
        
        # Predict Length
        length_pred = self.model_length.predict(X)[0]
        length_conf = max(self.model_length.predict_proba(X)[0])
        
        # Predict Shot
        shot_pred = self.model_shot.predict(X)[0]
        shot_conf = max(self.model_shot.predict_proba(X)[0])
        
        # Basic Regex to extract Bowler and Batter
        bowler = "Unknown"
        batter = "Unknown"
        
        # Format 1: HF Dataset "bowler name is X batsman name is Y"
        match_hf = re.search(r"bowler name is (.*?) batsman name is (.*?) over", text, re.IGNORECASE)
        # Format 2: UI Sample "Starc bowls... Kohli attempts"
        match_ui = re.search(r"^([A-Za-z\s\-]+)\s+bowls.*?,\s*([A-Za-z\s\-]+)\s+attempts", text, re.IGNORECASE)
        # Format 3: Classic "Starc to Kohli"
        match_classic = re.search(r"^([A-Za-z\s\-]+)\s+to\s+([A-Za-z\s\-]+),", text, re.IGNORECASE)
        
        if match_hf:
            bowler = match_hf.group(1).strip()
            batter = match_hf.group(2).strip()
        elif match_ui:
            bowler = match_ui.group(1).strip()
            batter = match_ui.group(2).strip()
        elif match_classic:
            bowler = match_classic.group(1).strip()
            batter = match_classic.group(2).strip()

        # Outcome extraction based on keyword heuristics
        outcome = "Unknown"
        text_lower = text.lower()
        if "dismissal is true" in text_lower or "out" in text_lower or "dismissal" in text_lower or "caught" in text_lower or "bowled" in text_lower or "lbw" in text_lower:
            outcome = "Wicket"
        elif "six" in text_lower or "6 runs" in text_lower:
            outcome = "Six"
        elif "four" in text_lower or "4 runs" in text_lower:
            outcome = "Four"
        elif "single" in text_lower or "1 run" in text_lower:
            outcome = "Single"
        elif "two runs" in text_lower or "2 runs" in text_lower:
            outcome = "Two"
        elif "three runs" in text_lower or "3 runs" in text_lower:
            outcome = "Three"
        elif "no run" in text_lower or "dot" in text_lower:
            outcome = "Dot Ball"
        elif "wide" in text_lower:
            outcome = "Wide"
        elif "no ball" in text_lower:
            outcome = "No Ball"

        # Fallback to "Unknown" if confidence is too low (e.g., < 40%)
        return {
            'line': f"{line_pred}" if line_conf > 0.4 else "Unknown",
            'length': f"{length_pred}" if length_conf > 0.4 else "Unknown",
            'shot': f"{shot_pred}" if shot_conf > 0.4 else "Unknown",
            'bowler': bowler,
            'batter': batter,
            'outcome': outcome
        }

if __name__ == "__main__":
    # Test block
    parser = CommentaryParser()
    test_str = "Starc bowls a full delivery outside off-stump, Kohli attempts a drive but edges it to first slip for a dismissal"
    print(parser.extract_features(test_str))
