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
        
    def encode_categorical(self, value, encoder):
        return encoder.transform([[value]])[0].tolist()

    def preprocess_sequence(self, sequence_data, phase, style, norm_sr, dismissal_rate, 
                            bowler_phase_econ, bowler_phase_wickets, bowler_type_avg, bowler_career_wickets, 
                            bowler_career_econ, bowler_career_avg, bowler_career_sr,
                            batsman_avg_vs_style, bowler_econ_vs_style, bowler_sr_vs_style, bowler_wkts_vs_style):
        """
        Converts a list of dicts (deliveries) into a PyTorch-ready tensor.
        Extracts categorical features and scales continuous variables.
        
        Args:
            sequence_data: List of dicts `[{'run': 1, 'length': 'Short'}, ...]`
            phase: Current Match Phase (e.g., 'Powerplay')
            style: Current Bowler Style (e.g., 'Pace')
            norm_sr: Scaled batsman strike rate (historical)
            dismissal_rate: Scaled batsman dismissal rate (historical)
            bowler_phase_econ: Bowler's economy in the current phase
            bowler_phase_wickets: Bowler's wickets in the current phase
            bowler_type_avg: Bowler's average against the current batsman type
            bowler_career_wickets: Bowler's total career wickets
            bowler_career_econ: Bowler's total career economy
            bowler_career_avg: Bowler's total career average
            bowler_career_sr: Bowler's total career strike rate
            batsman_avg_vs_style: Batsman's average against this bowler's style
            bowler_econ_vs_style: Bowler's economy against this batsman's style
            bowler_sr_vs_style: Bowler's strike rate against this batsman's style
            bowler_wkts_vs_style: Bowler's total wickets against this batsman's style
            
        Returns:
            torch.Tensor of shape (1, seq_length, 25)
        """
        # Normalization constraints
        norm_b_phase_econ = min(float(bowler_phase_econ) / 15.0, 1.0)
        norm_b_phase_wkts = min(float(bowler_phase_wickets) / 50.0, 1.0)
        
        # Handle 'N/A' or missing averages
        try:
            b_avg_float = float(bowler_type_avg)
            norm_b_type_avg = min(b_avg_float / 50.0, 1.0)
        except ValueError:
            norm_b_type_avg = 0.5 # Default middle ground for unknown
            
        norm_b_wkts = min(float(bowler_career_wickets) / 200.0, 1.0)
        norm_b_career_econ = min(float(bowler_career_econ) / 12.0, 1.0)
        
        try:
            c_avg_float = float(bowler_career_avg)
            norm_b_career_avg = min(c_avg_float / 50.0, 1.0)
        except ValueError:
            norm_b_career_avg = 0.5
            
        try:
            c_sr_float = float(bowler_career_sr)
            norm_b_career_sr = min(c_sr_float / 30.0, 1.0)
        except ValueError:
            norm_b_career_sr = 0.5
            
        # Normalization for the 4 new specific dimensions
        try:
            bat_avg_float = float(batsman_avg_vs_style)
            norm_bat_avg_vs_style = min(bat_avg_float / 60.0, 1.0)
        except ValueError:
            norm_bat_avg_vs_style = 0.5
            
        norm_b_econ_vs_style = min(float(bowler_econ_vs_style) / 15.0, 1.0)
        
        try:
            b_sr_float = float(bowler_sr_vs_style)
            norm_b_sr_vs_style = min(b_sr_float / 30.0, 1.0)
        except ValueError:
            norm_b_sr_vs_style = 0.5
            
        norm_b_wkts_vs_style = min(float(bowler_wkts_vs_style) / 100.0, 1.0)
            
        # One-hot context vectors
        phase_vec = self.encode_categorical(phase, self.encoder_phase)
        style_vec = self.encode_categorical(style, self.encoder_style)
        
        # Sequence construction
        sequence_vectors = []
        for delivery in sequence_data:
            # Scale Run [0, 6] -> [0, 1]
            run_scaled = min(delivery.get('run', 0) / 6.0, 1.0)
            
            # Encode Length
            len_vec = self.encode_categorical(delivery.get('length', ''), self.encoder_len)
            
            # Build the 25-Dimensional delivery vector
            # Vector Structure:
            # [0]   : Runs scaled
            # [1:6] : Length One-Hot (5)
            # [6:9] : Phase One-Hot (3)
            # [9:12]: Style One-Hot (3)
            # [12]  : Batsman Norm SR
            # [13]  : Batsman Dismissal Rate
            # [14]  : Bowler Phase Economy
            # [15]  : Bowler vs Bat Type Average
            # [16]  : Bowler Career Wickets
            # [17]  : Bowler Career Economy
            # [18]  : Bowler Career Average
            # [19]  : Bowler Career Strike Rate
            # [20]  : Batsman Average vs Current Bowler Style
            # [21]  : Bowler Economy vs Current Batsman Style
            # [22]  : Bowler Strike Rate vs Current Batsman Style
            # [23]  : Bowler Wickets vs Current Batsman Style
            # [24]  : Bowler Phase Wickets
            
            vector = [run_scaled] + len_vec + phase_vec + style_vec + [
                norm_sr, dismissal_rate, 
                norm_b_phase_econ, norm_b_type_avg, 
                norm_b_wkts, norm_b_career_econ, norm_b_career_avg, norm_b_career_sr,
                norm_bat_avg_vs_style, norm_b_econ_vs_style, norm_b_sr_vs_style, norm_b_wkts_vs_style,
                norm_b_phase_wkts
            ]
            sequence_vectors.append(vector)
            
        # Convert to numpy array then tensor
        tensor_data = np.array([sequence_vectors], dtype=np.float32)
        
        if TORCH_AVAILABLE:
            return torch.tensor(tensor_data)
        else:
            return tensor_data
