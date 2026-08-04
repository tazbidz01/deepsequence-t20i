import numpy as np
from sklearn.preprocessing import OneHotEncoder

try:
    import torch
    TORCH_AVAILABLE = True
except Exception as e:
    import traceback
    print(f"CRITICAL PYTORCH IMPORT ERROR (features): {e}")
    traceback.print_exc()
    TORCH_AVAILABLE = False

class SequencePreprocessor:
    def __init__(self):
        # We will use simple one-hot encoding for the categorical features
        self.length_categories = ["Yorker", "Full", "Slot", "Good Length", "Short"]
        self.phase_categories = ["Powerplay", "Middle Overs", "Death Overs"]
        self.style_categories = ["Pace", "Off-spin", "Leg-spin"]
        
        self.encoder_len = OneHotEncoder(categories=[self.length_categories], sparse_output=False, handle_unknown='ignore')
        self.encoder_phase = OneHotEncoder(categories=[self.phase_categories], sparse_output=False, handle_unknown='ignore')
        self.encoder_style = OneHotEncoder(categories=[self.style_categories], sparse_output=False, handle_unknown='ignore')
        
        # Fit once to initialize
        self.encoder_len.fit([["Yorker"], ["Full"], ["Slot"], ["Good Length"], ["Short"]])
        self.encoder_phase.fit([["Powerplay"], ["Middle Overs"], ["Death Overs"]])
        self.encoder_style.fit([["Pace"], ["Off-spin"], ["Leg-spin"]])
        
    def preprocess_sequence(self, sequence_data, match_phase, bowler_style, hist_sr=0.0, hist_dismissals=0, hist_balls=1):
        """
        Takes a list of dictionaries, e.g.:
        [{'run': 1, 'length': 'Good Length'}, ...]
        Returns a torch Tensor (or numpy array if PyTorch is broken) of shape (1, seq, 14)
        """
        # Encode global context (broadcasted to all timesteps)
        encoded_phase = self.encoder_phase.transform([[match_phase]])[0]
        encoded_style = self.encoder_style.transform([[bowler_style]])[0]
        # Calculate historical indices
        norm_sr = min(hist_sr / 200.0, 1.0) # Normalize against 200 SR
        hist_balls = max(hist_balls, 1) # Prevent division by zero
        dismissal_rate = float(hist_dismissals) / float(hist_balls)
        
        features = []
        for ball in sequence_data:
            # Normalize runs (divide by 6.0)
            norm_run = float(ball['run']) / 6.0
            
            # One-hot encode length
            length_val = ball['length']
            encoded_length = self.encoder_len.transform([[length_val]])[0]
            
            # Combine features: 1 (run) + 5 (len) + 3 (phase) + 3 (style) + 2 (historical) = 14 dimensions!
            ball_features = np.concatenate(([norm_run], encoded_length, encoded_phase, encoded_style, [norm_sr, dismissal_rate]))
            features.append(ball_features)
            
        np_features = np.array(features, dtype=np.float32)
        # Add batch dimension
        np_features = np.expand_dims(np_features, axis=0)
        
        if TORCH_AVAILABLE:
            return torch.tensor(np_features)
        else:
            return np_features
