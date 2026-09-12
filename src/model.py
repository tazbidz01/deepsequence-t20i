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
        def __init__(self, input_size=59, hidden_size=64, num_layers=2):
            super(DeepSequenceModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                batch_first=True,
                dropout=0.2
            )
            
            # Attention mechanism
            self.attention = nn.Linear(hidden_size, 1)
            
            # Fully connected layers for the final prediction
            self.fc1 = nn.Linear(hidden_size, 32)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.fc2 = nn.Linear(32, 1)
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
        def __init__(self, val, input_data=None):
            self.val = val
            self.input_data = input_data
            
        def item(self):
            # 46-Dimensional Baseline Calculation
            base_risk = 0.15
            
            if self.input_data is not None:
                try:
                    last_delivery = self.input_data[0, -1, :]
                    # Extracted Features (Pre-scaled 0.0 to 1.0)
                    # Index 25 is phase-specific SR. Index 50, 51 are true lifetime KPIs.
                    try:
                        bat_career_avg = float(last_delivery[50])
                        bat_career_sr = float(last_delivery[51])
                    except IndexError:
                        bat_career_avg, bat_career_sr = 0.5, 0.5
                    bat_avg_vs_style = float(last_delivery[33])
                    dismissal_rate = float(last_delivery[26])
                    
                    b_career_econ = float(last_delivery[30])
                    b_career_avg = float(last_delivery[31])
                    b_career_sr = float(last_delivery[32])
                    b_career_wkts = float(last_delivery[29])
                    
                    b_wkts_vs_style = float(last_delivery[36])
                    phase_wkts = float(last_delivery[37])
                    
                    # Match State Metrics
                    nstr_runs = float(last_delivery[40])
                    nstr_balls = float(last_delivery[41])
                    str_runs = float(last_delivery[38])
                    str_balls = float(last_delivery[39])
                    tb_runs = float(last_delivery[42])
                    tb_wkts = float(last_delivery[43])
                    sb_wkts = float(last_delivery[45])
                    
                    try:
                        sb_career_wkts = float(last_delivery[46])
                        sb_career_sr = float(last_delivery[49])
                    except IndexError:
                        sb_career_wkts, sb_career_sr = 0.0, 1.0
                        
                    # 52D to 56D: new match states
                    try:
                        partnership_strength = float(last_delivery[52])
                        crr = float(last_delivery[53])
                        wickets_fallen = float(last_delivery[54])
                        rrr = float(last_delivery[55])
                    except IndexError:
                        partnership_strength, crr, wickets_fallen, rrr = 0.5, 0.5, 0.0, 0.5
                        
                    # Calculate live match momentum modifiers based purely on Live Strike Rate
                    # Baseline T20 SR is 120.0. Deviations from this baseline dynamically scale risk.
                    nstr_pressure = 0.0
                    if nstr_balls > 0.1:
                        nstr_live_sr = (nstr_runs / nstr_balls) * 100.0
                        nstr_pressure = max(0.0, (120.0 - nstr_live_sr) * 0.001)
                        
                    striker_momentum = 0.0
                    if str_balls > 0.1:
                        str_live_sr = (str_runs / str_balls) * 100.0
                        striker_momentum = (120.0 - str_live_sr) * 0.002
                        
                    # ---------------------------------------------------------
                    # THE UNIFIED CONTINUOUS VULNERABILITY FORMULA (Match Context)
                    # ---------------------------------------------------------
                    base_risk = (
                        0.05 +
                        (1.0 - bat_career_sr) * 0.05 +
                        (1.0 - bat_career_avg) * 0.08 +
                        (dismissal_rate) * 0.05 +
                        (1.0 - bat_avg_vs_style) * 0.10 +
                        (1.0 - b_career_avg) * 0.08 +
                        (1.0 - b_career_econ) * 0.05 +
                        (1.0 - b_career_sr) * 0.05 +
                        (b_career_wkts) * 0.05 +
                        (b_wkts_vs_style) * 0.05 +
                        (phase_wkts) * 0.05 +
                        (nstr_pressure) * 0.10 +
                        (striker_momentum) +
                        (tb_wkts) * 0.10 - (tb_runs) * 0.05 +
                        (sb_wkts) * 0.05 +
                        (sb_career_wkts) * 0.05 +
                        (1.0 - sb_career_sr) * 0.05 +
                        (1.0 - partnership_strength) * 0.05 + # Weak partnerships = higher risk
                        (rrr - crr) * 0.10 + # RRR higher than CRR increases risk
                        (wickets_fallen) * 0.05
                    )
                    
                    risk = base_risk
                    
                    # ---------------------------------------------------------
                    # EVALUATE THE ACTUAL SEQUENCE EVENTS (First 25 Dimensions)
                    # ---------------------------------------------------------
                    seq = self.input_data[0]
                    total_runs_scaled = 0.0
                    death_yorkers = 0
                    
                    for ball in seq:
                        run_scaled = float(ball[0])
                        len_yorker = float(ball[1]) # Length index 1 is Yorker
                        phase_death = float(ball[21]) # Phase index 21 is Death Overs
                        
                        total_runs_scaled += run_scaled
                        if len_yorker == 1.0 and phase_death == 1.0:
                            death_yorkers += 1
                            
                    # Calculate total runs in the sequence (run_scaled is runs / 6.0)
                    total_runs = total_runs_scaled * 6.0
                    seq_len = len(seq)
                    
                    # Average runs per ball in the current sequence
                    runs_per_ball = total_runs / seq_len if seq_len > 0 else 0
                    
                    # High pressure if scoring less than 1 run a ball
                    # Continuous Sequence Momentum Adjustment
                    if runs_per_ball < 1.0:
                        # Penalty for scoring under 6 runs an over (Max penalty +0.25 at 0 runs)
                        risk += (1.0 - runs_per_ball) * 0.25
                    else:
                        # Reward for scoring over 6 runs an over (e.g. 12 runs = -0.25 risk)
                        risk -= (runs_per_ball - 1.0) * 0.25
                        
                    # Extra penalty for death yorkers
                    risk += (0.05 * death_yorkers)
                    
                    print(f"[DEBUG] base_risk={base_risk:.3f}, runs_per_ball={runs_per_ball:.2f}, risk_after_seq={risk:.3f}")
                    
                    return float(min(max(risk, 0.0), 1.0))
                        
                except Exception as e:
                    print(f"Mock heuristic error: {e}")
                    pass
                    
            return float(min(max(base_risk, 0.0), 1.0))

    class DeepSequenceModel:
        def __init__(self, input_size=59, hidden_size=64, num_layers=2):
            self.input_size = input_size
            
        def eval(self):
            pass
            
        def __call__(self, x):
            return MockTensor(0, input_data=x)

import os

# Singleton mock model for the UI
_mock_model = None

def get_model():
    global _mock_model
    if _mock_model is None:
        _mock_model = DeepSequenceModel(input_size=59, hidden_size=64, num_layers=2)
        
        # Load the trained PyTorch weights if they exist
        if TORCH_AVAILABLE:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "deepsequence_v1.6-baseline.pth")
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                    
                    # Dynamically pad the input weights if upgrading from 52D to 56D
                    if 'lstm.weight_ih_l0' in checkpoint:
                        w = checkpoint['lstm.weight_ih_l0']
                        if w.shape[1] == 52:
                            pad = torch.zeros(w.shape[0], 4).to(w.device)
                            checkpoint['lstm.weight_ih_l0'] = torch.cat([w, pad], dim=1)
                            
                    _mock_model.load_state_dict(checkpoint, strict=False)
                    _mock_model.eval()
                except Exception as e:
                    print(f"Error loading PyTorch weights: {e}")
                    
    return _mock_model
