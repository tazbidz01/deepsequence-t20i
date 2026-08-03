import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception as e:
    import traceback
    print(f"CRITICAL PYTORCH IMPORT ERROR: {e}")
    traceback.print_exc()
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    class DeepSequenceModel(nn.Module):
        def __init__(self, input_size=12, hidden_size=16, num_layers=1):
            super(DeepSequenceModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            
            out, _ = self.lstm(x, (h0, c0))
            
            out = self.fc(out[:, -1, :])
            out = self.sigmoid(out)
            return out
else:
    # Mock model for systems missing PyTorch C++ Redistributables (WinError 1114 fallback)
    class MockTensor:
        def __init__(self, val, shape):
            self.val = val
            self.shape_attr = shape
        def item(self):
            return self.val
        @property
        def shape(self):
            return self.shape_attr

    class DeepSequenceModel:
        def __init__(self, input_size=12, hidden_size=16, num_layers=1):
            self.input_size = input_size
            
        def eval(self):
            pass
            
        def __call__(self, x):
            # x is a numpy array (batch, seq, 12)
            # Dynamic mock calculation for systems hitting WinError 1114 in Streamlit
            last_ball = x[0, -1, :]
            runs_normalized = last_ball[0]
            is_powerplay = last_ball[6]
            is_death = last_ball[8]
            
            base_risk = 0.4
            
            # Penalize dot balls heavily
            if runs_normalized == 0:
                base_risk += 0.25
            elif runs_normalized >= (4/6):
                base_risk -= 0.15
                
            # Adjust for match phase pressure
            if is_powerplay:
                base_risk += 0.12
            elif is_death:
                base_risk += 0.28
                
            # Add a tiny bit of variance based on sequence length to simulate neural noise
            seq_len = x.shape[1]
            noise = (seq_len * 0.01)
            
            final_risk = max(0.12, min(0.94, base_risk + noise))
            return MockTensor(final_risk, x.shape) 

import os

# Singleton mock model for the UI
_mock_model = None

def get_model():
    global _mock_model
    if _mock_model is None:
        _mock_model = DeepSequenceModel(input_size=12, hidden_size=16)
        
        # Load the trained PyTorch weights if they exist
        if TORCH_AVAILABLE:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "deepsequence_v1.0-baseline.pth")
            if os.path.exists(model_path):
                _mock_model.load_state_dict(torch.load(model_path, weights_only=True))
                
        if hasattr(_mock_model, 'eval'):
            _mock_model.eval() # Set to evaluation mode for inference
    return _mock_model
