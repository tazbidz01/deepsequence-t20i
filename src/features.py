import numpy as np
from sklearn.preprocessing import OneHotEncoder

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

class SequencePreprocessor:
    def __init__(self):
        # We will use simple one-hot encoding for the "length" feature
        self.length_categories = ["Yorker", "Full", "Slot", "Good Length", "Short"]
        self.encoder = OneHotEncoder(categories=[self.length_categories], sparse_output=False, handle_unknown='ignore')
        # Fit once to initialize
        self.encoder.fit([["Yorker"], ["Full"], ["Slot"], ["Good Length"], ["Short"]])
        
    def preprocess_sequence(self, sequence_data):
        """
        Takes a list of dictionaries, e.g.:
        [{'run': 1, 'length': 'Good Length'}, ...]
        Returns a torch Tensor (or numpy array if PyTorch is broken)
        """
        features = []
        for ball in sequence_data:
            # Normalize runs (divide by 6.0)
            norm_run = float(ball['run']) / 6.0
            
            # One-hot encode length
            length_val = ball['length']
            encoded_length = self.encoder.transform([[length_val]])[0]
            
            # Combine features: 1 (run) + 5 (length) = 6 features per timestep
            ball_features = np.concatenate(([norm_run], encoded_length))
            features.append(ball_features)
            
        np_features = np.array(features, dtype=np.float32)
        # Add batch dimension
        np_features = np.expand_dims(np_features, axis=0)
        
        if TORCH_AVAILABLE:
            return torch.tensor(np_features)
        else:
            return np_features
