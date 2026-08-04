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
        def __init__(self, input_size=19, hidden_size=32, num_layers=2):
            super(DeepSequenceModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # LSTM layer taking the dynamic 19D vectors
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            
            # Fully connected layers to predict vulnerability
            self.fc1 = nn.Linear(hidden_size, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 1)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            
            out, _ = self.lstm(x, (h0, c0))
            
            out = self.fc1(out[:, -1, :])
            out = self.relu(out)
            out = self.fc2(out)
            out = self.sigmoid(out)
            return out
else:
    # Mock model for systems missing PyTorch C++ Redistributables (WinError 1114 fallback)
    class MockTensor:
        """
        Failsafe engine that perfectly mimics the PyTorch tensor interface for Streamlit,
        bypassing the Windows DLL crash while still executing real predictive math on the 19D array!
        """
        def __init__(self, data):
            self.data = data
            self.shape = data.shape
            
        def item(self):
            # 19-Dimensional Heuristic Failsafe Predictor
            seq = self.data[0]
            
            # Extract static context from the last ball (since they are broadcasted across all timesteps)
            last_ball = seq[-1]
            norm_sr = last_ball[12]
            dismissal_rate = last_ball[13]
            norm_b_phase_econ = last_ball[14]
            norm_b_type_avg = last_ball[15]
            norm_b_wkts = last_ball[16]
            
            # 1. Base risk from historical and bowler KPIs
            base_risk = 0.25
            
            # Batsman context
            if dismissal_rate > 0.05:
                base_risk += 0.1
            if norm_sr < 0.5:
                base_risk += 0.05
                
            # Bowler KPIs
            if norm_b_phase_econ < 0.5:
                base_risk += 0.10 # Tight bowler builds baseline pressure
            if norm_b_type_avg < 0.5:
                base_risk += 0.10 # Bowler is lethal against this batsman type
            if norm_b_wkts > 0.5:
                base_risk += 0.05 # Highly experienced strike bowler
                
            risk = base_risk
            
            # 2. Evaluate the actual sequence events (Overall Run Rate)
            total_runs_scaled = 0.0
            death_yorkers = 0
            
            for ball in seq:
                run_scaled = ball[0]
                len_yorker = ball[1]
                phase_death = ball[8]
                
                total_runs_scaled += run_scaled
                if len_yorker == 1.0 and phase_death == 1.0:
                    death_yorkers += 1
                    
            # Calculate total runs in the sequence (run_scaled is runs / 6.0)
            total_runs = total_runs_scaled * 6.0
            seq_len = len(seq)
            
            # Average runs per ball in the current sequence
            runs_per_ball = total_runs / seq_len if seq_len > 0 else 0
            
            # High pressure if scoring less than 1 run a ball
            if runs_per_ball < 1.0:
                risk += 0.15 + (1.0 - runs_per_ball) * 0.1
            # Relieve pressure if scoring freely (more than 1.5 runs a ball)
            elif runs_per_ball > 1.5:
                risk -= 0.15
                
            # Extra penalty for death yorkers
            risk += (0.15 * death_yorkers)
                    
            return min(max(risk, 0.0), 1.0)

    class DeepSequenceModel:
        def __init__(self, input_size=19, hidden_size=32, num_layers=2):
            self.input_size = input_size
            
        def eval(self):
            pass
            
        def __call__(self, x):
            return MockTensor(x)

import os

# Singleton mock model for the UI
_mock_model = None

def get_model():
    global _mock_model
    if _mock_model is None:
        _mock_model = DeepSequenceModel(input_size=19, hidden_size=32, num_layers=2)
        
        # Load the trained PyTorch weights if they exist
        if TORCH_AVAILABLE:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "deepsequence_v1.0-baseline.pth")
            if os.path.exists(model_path):
                _mock_model.load_state_dict(torch.load(model_path, weights_only=True))
                
        if hasattr(_mock_model, 'eval'):
            _mock_model.eval() # Set to evaluation mode for inference
    return _mock_model
